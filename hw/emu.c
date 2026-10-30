#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef __int128 i128;
typedef unsigned __int128 u128;

typedef struct {
  int rows, cols, E;
  const int8_t *q;
  const uint16_t *s;
} Mat;

static int L, D, H, KV, HD, F, V, S, INV_D, HS, WB, L2E, REP;
static Mat *M;
static const int32_t *RSQRT, *EXP2, *SIG, *COS, *SIN;
static int8_t *KC, *VC, *XQ;
static int *KSH, *VSH, *DD;
static int64_t *D64, *X, *T, *QKV, *ATT, *GU, *MID, *Y, *SA, *SP;
static dispatch_queue_t Q;

static inline int64_t wrap48(i128 v) { return (int64_t)((uint64_t)v << 16) >> 16; }

static inline i128 rsr(i128 v, int n) { return n > 0 ? (v + ((i128)1 << (n - 1))) >> n : (i128)((u128)v << -n); }

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
  size_t rb = WB == 8 ? m->cols : m->cols / 2;
  for (int g = 0; g < G; g++) D64[g] = d[g];
  dispatch_apply((rows + chunk - 1) / chunk, Q, ^(size_t c) {
    int r1 = (int)(c + 1) * chunk < rows ? (int)(c + 1) * chunk : rows;
    for (int r = (int)c * chunk; r < r1; r++)
      y[r] = WB == 8 ? row(m->q + r * rb, m->s + (size_t)r * G, xq, d, D64, G, 1)
                     : row(m->q + r * rb, m->s + (size_t)r * G, xq, d, D64, G, 0);
  });
}

static void rms_factor(const int64_t *x, int n, int64_t *lf, int *e) {
  i128 ss = 0;
  for (int i = 0; i < n; i++) ss += (i128)x[i] * x[i];
  i128 mean = ((ss * INV_D) >> 32) + 43;
  int ee = (bitlen((u128)mean) - 1 - 16) >> 1;
  i128 mm = ee >= 0 ? mean >> (2 * ee) : mean << (-2 * ee);
  *lf = RSQRT[(int)(mm >> 10) - 64];
  *e = ee;
}

static void mv(const Mat *m, const int64_t *x, int rms, int64_t *out) {
  int ref = quant_mx(x, m->cols, XQ, DD), e = 0;
  int64_t lf = 1;
  matvec(m, XQ, DD, Y);
  if (rms) rms_factor(x, m->cols, &lf, &e);
  int n = m->E - ref + (rms ? 7 + e : 0);
  for (int r = 0; r < m->rows; r++) out[r] = wrap48(rsr((i128)Y[r] * lf, n));
}

static void rope(int64_t *v, int nh, int pos) {
  const int32_t *c = COS + (size_t)pos * (HD / 2), *s = SIN + (size_t)pos * (HD / 2);
  for (int h = 0; h < nh; h++)
    for (int i = 0; i < HD / 2; i++) {
      int64_t a = v[h * HD + 2 * i], b = v[h * HD + 2 * i + 1];
      v[h * HD + 2 * i] = wrap48(rsr((i128)a * c[i] - (i128)b * s[i], 14));
      v[h * HD + 2 * i + 1] = wrap48(rsr((i128)a * s[i] + (i128)b * c[i], 14));
    }
}

static inline size_t kv_at(int l, int t, int g) { return ((size_t)l * S + t) * KV + g; }

static void attention(int l, int pos, const int64_t *q, int64_t *out) {
  int n = pos + 1;
  dispatch_apply(H, Q, ^(size_t h) {
    int g = (int)h / REP, kmin = INT_MAX, vmin = INT_MAX;
    int8_t qq[256];
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
    i128 num[256], inv = ((i128)1 << 32) / P;
    memset(num, 0, sizeof num);
    for (int t = 0; t < n; t++) {
      i128 pt = (i128)p[t] << (VSH[kv_at(l, t, g)] - vmin);
      const int8_t *v = VC + kv_at(l, t, g) * HD;
      for (int j = 0; j < HD; j++) num[j] += pt * v[j];
    }
    for (int j = 0; j < HD; j++) out[h * HD + j] = wrap48(rsr(num[j] * inv, 32 - vmin));
  });
}

static inline int64_t sigmoid(int64_t g) {
  if (g >= (8LL << 16)) return 1 << 16;
  if (g < -(8LL << 16)) return 0;
  int64_t i = (g + (8LL << 16)) >> 10;
  return SIG[i > 1023 ? 1023 : i];
}

static int wq(const Mat *m, size_t r, int i) {
  if (WB == 8) return m->q[r * m->cols + i];
  int8_t b = m->q[r * (m->cols / 2) + (i / 32) * 16 + i % 16];
  return i % 32 < 16 ? (int8_t)(b << 4) >> 4 : b >> 4;
}

static int step(int tok, int pos) {
  const Mat *e = &M[0];
  for (int i = 0; i < D; i++) X[i] = wrap48(rsr((int64_t)wq(e, tok, i) * e->s[(size_t)tok * (D / 32) + i / 32], e->E - 16));
  for (int l = 0; l < L; l++) {
    const Mat *m = &M[1 + 4 * l];
    mv(&m[0], X, 1, QKV);
    rope(QKV, H, pos);
    rope(QKV + H * HD, KV, pos);
    for (int g = 0; g < KV; g++) {
      KSH[kv_at(l, pos, g)] = quant8(QKV + (H + g) * HD, HD, KC + kv_at(l, pos, g) * HD);
      VSH[kv_at(l, pos, g)] = quant8(QKV + (H + KV + g) * HD, HD, VC + kv_at(l, pos, g) * HD);
    }
    attention(l, pos, QKV, ATT);
    mv(&m[1], ATT, 0, T);
    for (int i = 0; i < D; i++) X[i] = wrap48((i128)X[i] + T[i]);
    mv(&m[2], X, 1, GU);
    for (int i = 0; i < F; i++) MID[i] = wrap48(rsr(rsr((i128)GU[i] * sigmoid(GU[i]), 16) * GU[F + i], 16));
    mv(&m[3], MID, 0, T);
    for (int i = 0; i < D; i++) X[i] = wrap48((i128)X[i] + T[i]);
  }
  const Mat *hm = &M[1 + 4 * L];
  quant_mx(X, D, XQ, DD);
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
  if (b == MAP_FAILED || memcmp(b, "IZFEMU1", 8)) { fprintf(stderr, "bad image %s\n", path); exit(1); }
  const int32_t *h = (const int32_t *)(b + 8);
  L = h[0], D = h[1], H = h[2], KV = h[3], HD = h[4], F = h[5], V = h[6], S = h[7], INV_D = h[8], HS = h[9], WB = h[10], L2E = h[11];
  REP = H / KV;
  int n = 2 + 4 * L;
  const int32_t *E = h + 12;
  RSQRT = E + n, EXP2 = RSQRT + 192, SIG = EXP2 + 256, COS = SIG + 1024, SIN = COS + (size_t)S * HD / 2;
  M = calloc(n, sizeof(Mat));
  int dims[4][2] = {{(H + 2 * KV) * HD, D}, {D, H * HD}, {2 * F, D}, {D, F}};
  M[0] = (Mat){V, D}, M[n - 1] = (Mat){V, D};
  for (int l = 0; l < L; l++)
    for (int k = 0; k < 4; k++) M[1 + 4 * l + k] = (Mat){dims[k][0], dims[k][1]};
  const uint8_t *p = (const uint8_t *)(SIN + (size_t)S * HD / 2);
  for (int i = 0; i < n; i++) {
    M[i].E = E[i];
    M[i].q = (const int8_t *)p;
    p += (size_t)M[i].rows * M[i].cols / (WB == 8 ? 1 : 2);
  }
  for (int i = 0; i < n; i++) {
    M[i].s = (const uint16_t *)p;
    p += (size_t)M[i].rows * M[i].cols / 32 * 2;
  }
  if (p != b + st.st_size) { fprintf(stderr, "image size mismatch\n"); exit(1); }
  size_t kv = (size_t)L * S * KV;
  KC = malloc(kv * HD), VC = malloc(kv * HD), KSH = malloc(kv * sizeof(int)), VSH = malloc(kv * sizeof(int));
  int big = V > 2 * F ? V : 2 * F;
  XQ = malloc(F > D ? F : D), DD = malloc(sizeof(int) * (F > D ? F : D) / 32), D64 = malloc(8 * (F > D ? F : D) / 32);
  X = malloc(8 * D), T = malloc(8 * D), QKV = malloc(8 * (H + 2 * KV) * HD), ATT = malloc(8 * H * HD);
  GU = malloc(8 * 2 * F), MID = malloc(8 * F), Y = malloc(8 * (size_t)big), SA = malloc(8 * (size_t)H * S), SP = malloc(8 * (size_t)H * S);
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
    int pos = 0, next = 0, made = 0, done = n_prompt < 1 || n_prompt >= S;
    while (!done) {
      next = step(tok[pos], pos);
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
