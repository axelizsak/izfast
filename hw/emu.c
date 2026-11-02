#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef __int128 i128;
typedef unsigned __int128 u128;

typedef struct {
  int rows, cols, E, w8;
  const int8_t *q;
  const uint16_t *s;
} Mat;

typedef struct {
  int lin, f0, f1, f2;
  const int16_t *conv, *A, *norm, *qn, *kn;
  const int32_t *dt;
} Layer;

static int L, D, H, KV, HD, RD, F, V, S, INV_D, HS, LS, WB, HB, L2E, EPS, HK, HV, DK, DV, K, INV_HD, INV_DV, C, NA, NL;
static Mat *M;
static Layer *LY;
static const int32_t *RSQRT, *EXP2, *SIG, *SOFTPLUS, *EHI, *ELO, *COS, *SIN;
static int8_t *KC, *VC, *XQ;
static int *KSH, *VSH, *DD;
static int64_t *D64, *X, *T, *Y1, *ATT, *GU, *MID, *Y, *SA, *SP, *ST, *CS;
static dispatch_queue_t Q;
static int HREF;

static inline int64_t wrap48(i128 v) { return (int64_t)((uint64_t)v << 16) >> 16; }

static inline i128 rsr(i128 v, int n) { return n > 0 ? (v + ((i128)1 << (n - 1))) >> n : (i128)((u128)v << -n); }

static inline int64_t r16(int64_t v) { return (v + 32768) >> 16; }

static inline int bitlen(u128 m) {
  uint64_t hi = (uint64_t)(m >> 64), lo = (uint64_t)m;
  return hi ? 128 - __builtin_clzll(hi) : lo ? 64 - __builtin_clzll(lo) : 0;
}

static inline int shfind(uint64_t m) {
  int sh = bitlen(m) - 7;
  if (sh <= 0) return 0;
  return ((m + (1ULL << (sh - 1))) >> sh) > 127 ? sh + 1 : sh;
}

static inline uint64_t mag(int64_t v) { return v < 0 ? -(uint64_t)v : (uint64_t)v; }

static inline int8_t clip127(i128 v) { return v > 127 ? 127 : v < -127 ? -127 : (int8_t)v; }

static int quant8(const int64_t *x, int n, int8_t *xq) {
  uint64_t m = 0;
  for (int i = 0; i < n; i++) m = mag(x[i]) > m ? mag(x[i]) : m;
  int sh = shfind(m);
  for (int i = 0; i < n; i++) xq[i] = clip127(rsr(x[i], sh));
  return sh;
}

static int quant_mx(const int64_t *x, int n, int8_t *xq, int *d) {
  int G = n / 32, lo = INT_MAX, hi = 0;
  for (int g = 0; g < G; g++) {
    uint64_t m = 0;
    for (int i = 0; i < 32; i++) m = mag(x[32 * g + i]) > m ? mag(x[32 * g + i]) : m;
    d[g] = shfind(m);
    lo = d[g] < lo ? d[g] : lo;
    hi = d[g] > hi ? d[g] : hi;
  }
  int ref = hi - 24 > lo ? hi - 24 : lo;
  for (int g = 0; g < G; g++) {
    int sh = d[g] > ref ? d[g] : ref;
    d[g] = sh - ref;
    for (int i = 0; i < 32; i++) xq[32 * g + i] = clip127(rsr(x[32 * g + i], sh));
  }
  return ref;
}

static inline int32x4_t dot8(const int8_t *w, const int8_t *x) {
  return vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w), vld1q_s8(x)), vld1q_s8(w + 16), vld1q_s8(x + 16));
}

static inline int32x4_t dot4(const int8_t *w, const int8_t *x) {
  int8x16_t b = vld1q_s8(w);
  return vdotq_s32(vdotq_s32(vdupq_n_s32(0), vshrq_n_s8(vshlq_n_s8(b, 4), 4), vld1q_s8(x)), vshrq_n_s8(b, 4), vld1q_s8(x + 16));
}

static inline __attribute__((always_inline)) int64_t row(const int8_t *w, const uint16_t *s, const int8_t *x, const int *d,
                                                         const int64_t *d64, int G, int w8) {
  int64x2_t a0 = vdupq_n_s64(0), a1 = a0;
  int g = 0;
#define DOT(k) (w8 ? dot8(w + 32 * (k), x + 32 * (k)) : dot4(w + 16 * (k), x + 32 * (k)))
  for (; g + 4 <= G; g += 4) {
    int32x4_t dots = vpaddq_s32(vpaddq_s32(DOT(g), DOT(g + 1)), vpaddq_s32(DOT(g + 2), DOT(g + 3)));
    int32x4_t sc = vreinterpretq_s32_u32(vmovl_u16(vld1_u16(s + g)));
    a0 = vaddq_s64(a0, vshlq_s64(vmull_s32(vget_low_s32(dots), vget_low_s32(sc)), vld1q_s64(d64 + g)));
    a1 = vaddq_s64(a1, vshlq_s64(vmull_high_s32(dots, sc), vld1q_s64(d64 + g + 2)));
  }
  uint64_t acc = (uint64_t)vaddvq_s64(a0) + (uint64_t)vaddvq_s64(a1);
  for (; g < G; g++) acc += (uint64_t)((int64_t)vaddvq_s32(DOT(g)) * s[g]) << d[g];
#undef DOT
  return (int64_t)acc;
}

static void matvec(const Mat *m, const int8_t *xq, const int *d, int64_t *y) {
  int G = m->cols / 32, rows = m->rows, chunk = 16;
  size_t rb = m->w8 ? m->cols : m->cols / 2;
  for (int g = 0; g < G; g++) D64[g] = d[g];
  dispatch_apply((rows + chunk - 1) / chunk, Q, ^(size_t c) {
    int r1 = (int)(c + 1) * chunk < rows ? (int)(c + 1) * chunk : rows;
    for (int r = (int)c * chunk; r < r1; r++)
      y[r] = m->w8 ? row(m->q + r * rb, m->s + (size_t)r * G, xq, d, D64, G, 1)
                   : row(m->q + r * rb, m->s + (size_t)r * G, xq, d, D64, G, 0);
  });
}

static void rms_factor(const int64_t *x, int n, int64_t inv, int64_t *lf, int *e) {
  i128 ss = 0;
  for (int i = 0; i < n; i++) ss += (i128)x[i] * x[i];
  i128 mean = ((ss * inv) >> 32) + EPS;
  int ee = (bitlen((u128)mean) - 1 - 16) >> 1;
  i128 mm = ee >= 0 ? mean >> (2 * ee) : mean << (-2 * ee);
  *lf = RSQRT[(int)(mm >> 10) - 64];
  *e = ee;
}

static void norm_apply(const int64_t *x, int n, int64_t inv, int64_t *out) {
  int64_t lf;
  int e;
  rms_factor(x, n, inv, &lf, &e);
  for (int i = 0; i < n; i++) out[i] = wrap48(rsr((i128)x[i] * lf, 7 + e));
}

static void mv(const Mat *m, const int64_t *x, int rms, int64_t *out) {
  int ref = quant_mx(x, m->cols, XQ, DD), e = 0;
  int64_t lf = 1;
  matvec(m, XQ, DD, Y);
  if (rms) rms_factor(x, m->cols, INV_D, &lf, &e);
  int n = m->E - ref + (rms ? 7 + e : 0);
  for (int r = 0; r < m->rows; r++) out[r] = wrap48(rsr((i128)Y[r] * lf, n));
}

static inline int64_t sigmoid(int64_t g) {
  if (g >= (8LL << 16)) return 1 << 16;
  if (g < -(8LL << 16)) return 0;
  int64_t i = (g + (8LL << 16)) >> 10;
  return SIG[i > 1023 ? 1023 : i];
}

static inline int64_t silu(int64_t x) { return (int64_t)rsr((i128)x * sigmoid(x), 16); }

static inline int64_t softplus(int64_t x) {
  if (x >= (8LL << 16)) return x;
  if (x < -(8LL << 16)) return 0;
  int64_t i = (x + (8LL << 16)) >> 8;
  return SOFTPLUS[i > 4095 ? 4095 : i];
}

static inline int64_t exp2_neg(int64_t u) {
  int64_t v = ((int64_t)EHI[(u >> 8) & 255] * ELO[u & 255] + (1 << 15)) >> 16;
  return (u >> 16) >= 17 ? 0 : v >> (u >> 16);
}

static void rope(int64_t *v, int pos) {
  const int32_t *c = COS + (size_t)pos * (RD / 2), *s = SIN + (size_t)pos * (RD / 2);
  for (int i = 0; i < RD / 2; i++) {
    int64_t a = v[2 * i], b = v[2 * i + 1];
    v[2 * i] = wrap48(rsr((i128)a * c[i] - (i128)b * s[i], 14));
    v[2 * i + 1] = wrap48(rsr((i128)a * s[i] + (i128)b * c[i], 14));
  }
}

static inline size_t kv_at(int l, int t, int g) { return ((size_t)l * S + t) * KV + g; }

static void attention(int l, int pos, const int64_t *q, int64_t *out) {
  int n = pos + 1, rep = H / KV;
  dispatch_apply(H, Q, ^(size_t h) {
    int g = (int)h / rep, kmin = INT_MAX, vmin = INT_MAX;
    int8_t qq[512];
    int qsh = quant8(q + h * HD, HD, qq);
    int64_t *a = SA + h * S, *p = SP + h * S, amax = INT64_MIN, P = 0;
    for (int t = 0; t < n; t++) {
      kmin = KSH[kv_at(l, t, g)] < kmin ? KSH[kv_at(l, t, g)] : kmin;
      vmin = VSH[kv_at(l, t, g)] < vmin ? VSH[kv_at(l, t, g)] : vmin;
    }
    for (int t = 0; t < n; t++) {
      const int8_t *k = KC + kv_at(l, t, g) * HD;
      int32_t dot = 0;
      for (int j = 0; j < HD; j++) dot += k[j] * qq[j];
      a[t] = (int64_t)((uint64_t)(int64_t)dot << (KSH[kv_at(l, t, g)] - kmin));
      amax = a[t] > amax ? a[t] : amax;
    }
    i128 c = (i128)L2E * HS;
    int nsh = 54 - qsh - kmin;
    for (int t = 0; t < n; t++) {
      i128 u = rsr(((i128)amax - a[t]) * c, nsh), ip = u >> 8;
      p[t] = ip >= 17 ? 0 : EXP2[(int)(u & 255)] >> (int)ip;
      P += p[t];
    }
    i128 num[512], inv = ((i128)1 << 32) / P;
    memset(num, 0, sizeof num);
    for (int t = 0; t < n; t++) {
      i128 pt = (i128)p[t] << (VSH[kv_at(l, t, g)] - vmin);
      const int8_t *v = VC + kv_at(l, t, g) * HD;
      for (int j = 0; j < HD; j++) num[j] += pt * v[j];
    }
    for (int j = 0; j < HD; j++) out[h * HD + j] = wrap48(rsr(num[j] * inv, 32 - vmin));
  });
}

static void gated_attention(const Layer *ly, int ia, int pos, const int64_t *y, int64_t *out) {
  static int64_t q[16384], k[4096], tmp[512];
  for (int h = 0; h < H; h++) {
    norm_apply(y + h * HD, HD, INV_HD, tmp);
    for (int i = 0; i < HD; i++) q[h * HD + i] = wrap48(rsr((i128)tmp[i] * ly->qn[i], ly->f0));
    rope(q + h * HD, pos);
  }
  for (int g = 0; g < KV; g++) {
    norm_apply(y + (2 * H + g) * HD, HD, INV_HD, tmp);
    for (int i = 0; i < HD; i++) k[g * HD + i] = wrap48(rsr((i128)tmp[i] * ly->kn[i], ly->f1));
    rope(k + g * HD, pos);
    KSH[kv_at(ia, pos, g)] = quant8(k + g * HD, HD, KC + kv_at(ia, pos, g) * HD);
    VSH[kv_at(ia, pos, g)] = quant8(y + (2 * H + KV + g) * HD, HD, VC + kv_at(ia, pos, g) * HD);
  }
  attention(ia, pos, q, ATT);
  for (int i = 0; i < H * HD; i++) out[i] = wrap48(rsr((i128)ATT[i] * sigmoid(y[H * HD + i]), 16));
}

static void delta(const Layer *ly, int il, const int64_t *y, int64_t *out) {
  int64_t *cs = CS + (size_t)il * (K - 1) * C, *st = ST + (size_t)il * HV * DK * DV;
  static int64_t mx[16384], qn[4096], kn[4096], dec[256], beta[256], tmp[256];
  for (int c = 0; c < C; c++) {
    i128 acc = 0;
    for (int j = 0; j < K - 1; j++) acc += (i128)cs[(size_t)j * C + c] * ly->conv[c * K + j];
    acc += (i128)y[c] * ly->conv[c * K + K - 1];
    mx[c] = silu(wrap48(rsr(acc, ly->f0)));
    for (int j = 0; j < K - 2; j++) cs[(size_t)j * C + c] = cs[(size_t)(j + 1) * C + c];
    cs[(size_t)(K - 2) * C + c] = y[c];
  }
  int64_t one = (int64_t)1 << 32;
  for (int h = 0; h < HK; h++) {
    norm_apply(mx + h * DK, DK, one, tmp);
    for (int i = 0; i < DK; i++) qn[h * DK + i] = (int64_t)rsr((i128)tmp[i] * LS, 15);
    norm_apply(mx + HK * DK + h * DK, DK, one, kn + h * DK);
  }
  const int64_t *z = y + C, *b = z + HV * DV, *a = b + HV;
  for (int h = 0; h < HV; h++) {
    beta[h] = sigmoid(b[h]);
    int64_t sp = softplus(a[h] + ly->dt[h]);
    dec[h] = exp2_neg((int64_t)rsr((i128)sp * ly->A[h] * L2E, ly->f1 + 15));
  }
  int64_t top = ((int64_t)1 << 31) - 1;
  dispatch_apply(HV, Q, ^(size_t hh) {
    int h = (int)hh, r = HV / HK;
    const int64_t *kk = kn + (h / r) * DK, *qq = qn + (h / r) * DK, *v = mx + 2 * HK * DK + h * DV;
    int64_t *s = st + (size_t)h * DK * DV, kv[256], dl[256], o[256], on[256];
    for (int i = 0; i < DK * DV; i++) {
      int64_t t = r16(s[i] * dec[h]);
      s[i] = t > top ? top : t < -top ? -top : t;
    }
    for (int j = 0; j < DV; j++) {
      int64_t acc = 0;
      for (int i = 0; i < DK; i++) acc += kk[i] * s[i * DV + j];
      kv[j] = r16(acc);
      dl[j] = r16((v[j] - kv[j]) * beta[h]);
    }
    for (int i = 0; i < DK; i++)
      for (int j = 0; j < DV; j++) {
        int64_t t = s[i * DV + j] + r16(kk[i] * dl[j]);
        s[i * DV + j] = t > top ? top : t < -top ? -top : t;
      }
    for (int j = 0; j < DV; j++) {
      int64_t acc = 0;
      for (int i = 0; i < DK; i++) acc += qq[i] * s[i * DV + j];
      o[j] = r16(acc);
    }
    norm_apply(o, DV, INV_DV, on);
    for (int j = 0; j < DV; j++) {
      int64_t g = wrap48(rsr((i128)on[j] * ly->norm[j], ly->f2));
      out[h * DV + j] = wrap48(rsr((i128)g * silu(z[h * DV + j]), 16));
    }
  });
}

static int wq(const Mat *m, size_t r, int i) {
  if (m->w8) return m->q[r * m->cols + i];
  int8_t b = m->q[r * (m->cols / 2) + (i / 32) * 16 + i % 16];
  return i % 32 < 16 ? (int8_t)(b << 4) >> 4 : b >> 4;
}

static int step(int tok, int pos) {
  const Mat *e = &M[0];
  for (int i = 0; i < D; i++) X[i] = wrap48(rsr((int64_t)wq(e, tok, i) * e->s[(size_t)tok * (D / 32) + i / 32], e->E - 16));
  int il = 0, ia = 0;
  for (int l = 0; l < L; l++) {
    const Mat *m = &M[1 + 4 * l];
    const Layer *ly = &LY[l];
    mv(&m[0], X, 1, Y1);
    if (ly->lin) delta(ly, il++, Y1, ATT);
    else gated_attention(ly, ia++, pos, Y1, ATT);
    mv(&m[1], ATT, 0, T);
    for (int i = 0; i < D; i++) X[i] = wrap48((i128)X[i] + T[i]);
    mv(&m[2], X, 1, GU);
    for (int i = 0; i < F; i++) MID[i] = wrap48(rsr((i128)silu(GU[i]) * GU[F + i], 16));
    mv(&m[3], MID, 0, T);
    for (int i = 0; i < D; i++) X[i] = wrap48((i128)X[i] + T[i]);
  }
  const Mat *hm = &M[1 + 4 * L];
  HREF = quant_mx(X, D, XQ, DD);
  matvec(hm, XQ, DD, Y);
  int best = 0;
  for (int r = 1; r < hm->rows; r++) best = Y[r] > Y[best] ? r : best;
  return best;
}

static void load(const char *path) {
  int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st)) { perror(path); exit(1); }
  const uint8_t *b = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (b == MAP_FAILED || memcmp(b, "IZFQWEN1", 8)) { fprintf(stderr, "bad image %s\n", path); exit(1); }
  const int32_t *h = (const int32_t *)(b + 8);
  L = h[0], D = h[1], H = h[2], KV = h[3], HD = h[4], RD = h[5], F = h[6], V = h[7], S = h[8], INV_D = h[9], HS = h[10], LS = h[11];
  WB = h[12], HB = h[13], L2E = h[14], EPS = h[15], HK = h[16], HV = h[17], DK = h[18], DV = h[19], K = h[20], INV_HD = h[21], INV_DV = h[22];
  C = 2 * HK * DK + HV * DV;
  const int32_t *kinds = h + 23, *E = kinds + L, *fx = E + 2 + 4 * L;
  int n = 2 + 4 * L;
  RSQRT = fx + 3 * L, EXP2 = RSQRT + 192, SIG = EXP2 + 256, SOFTPLUS = SIG + 1024, EHI = SOFTPLUS + 4096, ELO = EHI + 256;
  COS = ELO + 256, SIN = COS + (size_t)S * RD / 2;
  const uint8_t *p = (const uint8_t *)(SIN + (size_t)S * RD / 2);
  LY = calloc(L, sizeof(Layer));
  M = calloc(n, sizeof(Mat));
  M[0] = (Mat){V, D}, M[1 + 4 * L] = (Mat){V, D};
  NA = NL = 0;
  for (int l = 0; l < L; l++) {
    Layer *ly = &LY[l];
    ly->lin = kinds[l], ly->f0 = fx[3 * l], ly->f1 = fx[3 * l + 1], ly->f2 = fx[3 * l + 2];
    if (ly->lin) {
      NL++;
      ly->conv = (const int16_t *)p, p += (size_t)C * K * 2;
      ly->dt = (const int32_t *)p, p += (size_t)HV * 4;
      ly->A = (const int16_t *)p, p += (size_t)HV * 2;
      ly->norm = (const int16_t *)p, p += (size_t)DV * 2;
      M[1 + 4 * l] = (Mat){C + HV * DV + 2 * HV, D}, M[2 + 4 * l] = (Mat){D, HV * DV};
    } else {
      NA++;
      ly->qn = (const int16_t *)p, p += (size_t)HD * 2;
      ly->kn = (const int16_t *)p, p += (size_t)HD * 2;
      M[1 + 4 * l] = (Mat){(2 * H + 2 * KV) * HD, D}, M[2 + 4 * l] = (Mat){D, H * HD};
    }
    M[3 + 4 * l] = (Mat){2 * F, D}, M[4 + 4 * l] = (Mat){D, F};
  }
  for (int i = 0; i < n; i++) {
    M[i].E = E[i];
    M[i].w8 = (i == 0 || i == n - 1 ? HB : WB) == 8;
    M[i].q = (const int8_t *)p;
    p += (size_t)M[i].rows * M[i].cols / (M[i].w8 ? 1 : 2);
  }
  for (int i = 0; i < n; i++) {
    M[i].s = (const uint16_t *)p;
    p += (size_t)M[i].rows * M[i].cols / 32 * 2;
  }
  if (p != b + st.st_size) { fprintf(stderr, "image size mismatch\n"); exit(1); }
  size_t kv = (size_t)(NA ? NA : 1) * S * KV;
  KC = malloc(kv * HD), VC = malloc(kv * HD), KSH = malloc(kv * sizeof(int)), VSH = malloc(kv * sizeof(int));
  int wide = C + HV * DV + 2 * HV > (2 * H + 2 * KV) * HD ? C + HV * DV + 2 * HV : (2 * H + 2 * KV) * HD;
  int big = V > 2 * F ? V : 2 * F, in = F > HV * DV ? F : HV * DV;
  in = in > H * HD ? in : H * HD, in = in > D ? in : D;
  XQ = malloc(in), DD = malloc(sizeof(int) * in / 32), D64 = malloc(8 * in / 32);
  X = malloc(8 * D), T = malloc(8 * D), Y1 = malloc(8 * (size_t)wide), ATT = malloc(8 * (size_t)in);
  GU = malloc(8 * 2 * F), MID = malloc(8 * F), Y = malloc(8 * (size_t)(big > wide ? big : wide));
  SA = malloc(8 * (size_t)H * S), SP = malloc(8 * (size_t)H * S);
  ST = calloc((size_t)(NL ? NL : 1) * HV * DK * DV, 8), CS = calloc((size_t)(NL ? NL : 1) * (K - 1) * C, 8);
}

static void reset(void) {
  memset(ST, 0, (size_t)(NL ? NL : 1) * HV * DK * DV * 8);
  memset(CS, 0, (size_t)(NL ? NL : 1) * (K - 1) * C * 8);
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: emu image.bin\n"); return 1; }
  load(argv[1]);
  Q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  printf("ready %d %d %d\n", L, D, S);
  fflush(stdout);
  static char line[1 << 20];
  int *tok = malloc(sizeof(int) * S), stop[64];
  while (fgets(line, sizeof line, stdin)) {
    char mode, *c = line;
    int n_new, n_prompt, n_stop, k;
    if (sscanf(c, " %c %d %d %d%n", &mode, &n_new, &n_prompt, &n_stop, &k) != 4) break;
    c += k;
    for (int i = 0; i < n_prompt && i < S; i++) tok[i] = (int)strtol(c, &c, 10);
    for (int i = 0; i < n_stop && i < 64; i++) stop[i] = (int)strtol(c, &c, 10);
    reset();
    int pos = 0, next = 0, made = 0, done = n_prompt < 1 || n_prompt >= S;
    while (!done) {
      next = step(tok[pos], pos);
      if (mode == 'l' && pos + 1 < n_prompt) {
        double ss = 0, mx = -INFINITY, sum = 0;
        for (int i = 0; i < D; i++) ss += ((double)X[i] / 65536) * ((double)X[i] / 65536);
        double sc = ldexp(1.0, HREF - M[1 + 4 * L].E - 16) / sqrt(ss / D + ldexp((double)EPS, -32));
        for (int r = 0; r < V; r++) mx = Y[r] * sc > mx ? Y[r] * sc : mx;
        for (int r = 0; r < V; r++) sum += exp(Y[r] * sc - mx);
        printf("l %d %.6f %d\n", pos, Y[tok[pos + 1]] * sc - mx - log(sum), next);
      }
      if (mode == 'x') {
        printf("x %d", pos);
        for (int i = 0; i < D; i++) printf(" %lld", (long long)X[i]);
        printf("\n");
      }
      if (++pos >= n_prompt) {
        printf("t %d\n", next);
        made++;
        for (int i = 0; i < n_stop; i++) done |= next == stop[i];
        done |= made >= n_new || pos >= S;
        if (!done) tok[pos] = next;
      }
      fflush(stdout);
    }
    printf("end\n");
    fflush(stdout);
  }
  return 0;
}
