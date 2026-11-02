module dnu #(
  parameter int VK   = 1,
  parameter int HK   = 2,
  parameter int HV   = 4,
  parameter int DK   = 32,
  parameter int DV   = 32,
  parameter int CK   = 4,
  parameter int NLIN = 3,
  parameter int SH   = 1,
  parameter int LS   = 5793,
  parameter int EPS  = 4
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               tbl_we,
  input  logic [3:0]         tbl_sel,
  input  logic [31:0]        tbl_addr,
  input  logic [31:0]        tbl_data,
  input  logic [31:0]        pos,
  input  logic               go,
  input  logic [31:0]        fa,
  input  logic [31:0]        fc,
  input  logic [31:0]        fk,
  input  logic signed [15:0] imm,
  output logic               done,
  output logic [31:0]        rd_row,
  input  logic [32*VK*48-1:0] rd_data,
  output logic               o_we,
  output logic [31:0]        o_row,
  output logic signed [47:0] o_data [32*VK],
  output logic [VK-1:0]      o_mask,
  output logic               o_first,
  output logic [31:0]        o_g0
);

  localparam int VW   = 32 * VK;
  localparam int C    = 2 * HK * DK + HV * DV;
  localparam int CR   = C / VW;
  localparam int KR   = DK / VW;
  localparam int VR   = DV / VW;
  localparam int RR   = HV / HK;
  localparam int NG   = HV / SH;
  localparam int PR   = 2 * KR + 2 * VR;
  localparam int SDEP = NLIN * NG * DK;
  localparam int SA   = $clog2(SDEP);
  localparam int L2E  = 47274;
  localparam int LDV  = $clog2(DV);
  localparam int SQW  = 104;
  localparam int RL   = ($clog2(VW) + 1) / 2;
  localparam int ABR  = ((C + HV * DV) % VW + 2 * HV + VW - 1) / VW;
  localparam logic signed [31:0] TOP = 32'sh7fffffff;

  typedef logic signed [47:0] v48;

  function automatic v48 rnd(input logic signed [127:0] v, input int n);
    if (n <= 0) rnd = 48'(v <<< (-n));
    else if (n >= 120) rnd = '0;
    else rnd = 48'((v + (128'sd1 <<< (n - 1))) >>> n);
  endfunction

  function automatic logic signed [31:0] clip32(input logic signed [71:0] v);
    clip32 = (v > 72'(TOP)) ? TOP : (v < -72'(TOP)) ? -TOP : 32'(v);
  endfunction

  function automatic logic signed [71:0] r16(input logic signed [71:0] v);
    r16 = (v + 72'sd32768) >>> 16;
  endfunction

  function automatic logic [1:0] mod3(input logic [31:0] x);
    logic [5:0] a;
    logic [3:0] b, c;
    a = '0;
    for (int k = 0; k < 16; k++) a = a + 6'(x[2*k +: 2]);
    b = 4'(a[1:0]) + 4'(a[3:2]) + 4'(a[5:4]);
    c = 4'(b[1:0]) + 4'(b[3:2]);
    mod3 = (c >= 4'd3) ? 2'(c - 4'd3) : 2'(c);
  endfunction

  function automatic int bitlen8(input logic [7:0] v);
    bitlen8 = 0;
    for (int k = 0; k < 8; k++) if (v[k]) bitlen8 = k + 1;
  endfunction

  logic [15:0] t_rsqrt [192];
  logic [16:0] t_sig [1024];
  logic [19:0] t_sp [4096];
  logic [16:0] t_hi [256], t_lo [256];
  logic signed [31:0] t_dt [NLIN*HV];
  logic signed [15:0] t_a [NLIN*HV];
  logic signed [15:0] t_ng [NLIN*DV];
  logic [VW*16-1:0] cw [CK][NLIN*CR];
  logic [VW*48-1:0] cs [CK-1][NLIN*CR];

  always_ff @(posedge clk) if (tbl_we) begin
    case (tbl_sel)
      4'd0: t_rsqrt[tbl_addr] <= tbl_data[15:0];
      4'd2: t_sig[tbl_addr] <= tbl_data[16:0];
      4'd6: t_sp[tbl_addr] <= tbl_data[19:0];
      4'd7: t_hi[tbl_addr] <= tbl_data[16:0];
      4'd8: t_lo[tbl_addr] <= tbl_data[16:0];
      4'd9: cw[tbl_addr % CK][(tbl_addr / CK) / VW][((tbl_addr / CK) % VW) * 16 +: 16] <= tbl_data[15:0];
      4'd10: t_dt[tbl_addr] <= tbl_data;
      4'd11: t_a[tbl_addr] <= tbl_data[15:0];
      4'd12: t_ng[tbl_addr] <= tbl_data[15:0];
      default: ;
    endcase
  end

  function automatic logic [9:0] sidx(input v48 g);
    sidx = 10'((g + 48'sd524288) >>> 10);
  endfunction

  function automatic logic [16:0] sfix(input v48 g, input logic [16:0] t);
    sfix = (g >= 48'sd524288) ? 17'd65536 : (g < -48'sd524288) ? 17'd0 : t;
  endfunction

  typedef enum logic [3:0] {D_IDLE, D_CONV, D_WAIT, D_AB, D_GATE, D_LOAD, D_NF, D_SC, D_P1, D_DL, D_P2, D_OS, D_OF, D_OUT, D_DRAIN} dstate_t;
  dstate_t st;
  int il, cnt, gi, wcnt;
  logic [7:0] af, gf, cf;
  logic signed [31:0] cpos;
  logic [1:0] cmod;
  logic [31:0] base, dst;

  logic signed [47:0] breg [HV], areg [HV];
  logic [16:0] beta [HV], dec [HV];
  v48 qr [SH][DK], kr [SH][DK], vr [SH][DV], zr [SH][DV];
  logic signed [18:0] qn [SH][DK], kn [SH][DK];
  logic signed [63:0] kva [SH][DV], oa [SH][DV];
  logic signed [49:0] dl [SH][DV];
  v48 ov [SH][DV];
  logic [SQW-1:0] ssa [3][SH];
  logic [15:0] fl [3][SH];
  logic signed [7:0] fe [3][SH];

  function automatic int ld_row(input logic [31:0] b, input int g, input int k);
    int s, part, h;
    s = k / PR;
    part = k % PR;
    h = g * SH + s;
    if (part < KR) ld_row = (int'(b) + (h / RR) * DK) / VW + part;
    else if (part < 2 * KR) ld_row = (int'(b) + HK * DK + (h / RR) * DK) / VW + part - KR;
    else if (part < 2 * KR + VR) ld_row = (int'(b) + 2 * HK * DK + h * DV) / VW + part - 2 * KR;
    else ld_row = (int'(b) + C + h * DV) / VW + part - 2 * KR - VR;
  endfunction

  always_comb begin
    if (st == D_CONV) rd_row = base / VW + 32'(cnt);
    else if (st == D_AB) rd_row = (base + 32'(C + HV * DV)) / VW + 32'(cnt);
    else rd_row = 32'(ld_row(base, gi, cnt));
  end

  logic c1_v, c2_v, c3_v, c4_v, c5_v;
  int   c1_r, c2_r, c3_r, c4_r, c5_r;
  logic [VW*48-1:0] c1_st [CK-1];
  logic [VW*48-1:0] c1_x;
  logic [VW*16-1:0] c1_w [CK];
  logic signed [63:0] c2_p [VW][CK];
  v48 c3_x [VW], c4_x [VW];
  logic [16:0] c4_s [VW];
  logic signed [64:0] c5_p [VW];
  logic cv_issue;
  int   cv_row;

  always_ff @(posedge clk) begin
    for (int j = 0; j < CK; j++) c1_w[j] <= cw[j][il * CR + cv_row];
    for (int j = 0; j < CK - 1; j++) c1_st[j] <= cs[j][il * CR + cv_row];
    c1_r <= cv_row;
    if (cv_issue) c1_x <= rd_data;
    c2_r <= c1_r; c3_r <= c2_r; c4_r <= c3_r; c5_r <= c4_r;
    if (c1_v) begin
      for (int u = 0; u < VW; u++) begin
        c2_p[u][CK-1] <= 64'($signed(c1_x[u*48 +: 48])) * 64'($signed(c1_w[CK-1][u*16 +: 16]));
        for (int j = 0; j < CK - 1; j++) begin
          int t;
          t = int'(cmod) + j;
          t = (t >= CK - 1) ? t - (CK - 1) : t;
          c2_p[u][j] <= (cpos < CK - 1 - j) ? 64'sd0 : 64'($signed(c1_st[t][u*48 +: 48])) * 64'($signed(c1_w[j][u*16 +: 16]));
        end
      end
      cs[cmod][il * CR + c1_r] <= c1_x;
    end
    if (c2_v)
      for (int u = 0; u < VW; u++) begin
        logic signed [127:0] a;
        a = '0;
        for (int j = 0; j < CK; j++) a = a + 128'(c2_p[u][j]);
        c3_x[u] <= rnd(a, int'(cf));
      end
    if (c3_v)
      for (int u = 0; u < VW; u++) begin
        c4_x[u] <= c3_x[u];
        c4_s[u] <= sfix(c3_x[u], t_sig[sidx(c3_x[u])]);
      end
    if (c4_v)
      for (int u = 0; u < VW; u++) c5_p[u] <= 65'(c4_x[u]) * 65'($signed({1'b0, c4_s[u]}));
  end

  logic g1_v, g2_v, g3_v, g4_v, g5_v, g6_v, g7_v;
  int   g1_h, g2_h, g3_h, g4_h, g5_h, g6_h, g7_h;
  logic signed [48:0] g1_x;
  v48 g2_sp;
  logic signed [63:0] g3_t;
  logic signed [79:0] g4_t;
  logic signed [63:0] g5_u;
  logic [16:0] g6_hi, g6_lo;
  logic [5:0]  g6_ip;
  logic        g6_z;
  logic [33:0] g7_v34;
  logic [5:0]  g7_ip;
  logic        g7_z;
  logic        ga_issue;
  int          ga_h;

  always_ff @(posedge clk) begin
    g1_h <= ga_h; g2_h <= g1_h; g3_h <= g2_h; g4_h <= g3_h; g5_h <= g4_h; g6_h <= g5_h; g7_h <= g6_h;
    if (ga_issue) begin
      g1_x <= 49'(areg[ga_h]) + 49'(t_dt[il * HV + ga_h]);
      beta[ga_h] <= sfix(breg[ga_h], t_sig[sidx(breg[ga_h])]);
    end
    if (g1_v) g2_sp <= (g1_x >= 49'sd524288) ? 48'(g1_x) : (g1_x < -49'sd524288) ? '0 : 48'(t_sp[12'((g1_x + 49'sd524288) >>> 8)]);
    if (g2_v) g3_t <= 64'(g2_sp) * 64'(t_a[il * HV + g2_h]);
    if (g3_v) g4_t <= 80'(g3_t) * 80'(L2E);
    if (g4_v) g5_u <= 64'(rnd(128'(g4_t), int'(af) + 15));
    if (g5_v) begin
      g6_hi <= t_hi[g5_u[15:8]];
      g6_lo <= t_lo[g5_u[7:0]];
      g6_z <= (g5_u >>> 16) >= 17;
      g6_ip <= g5_u[21:16];
    end
    if (g6_v) begin g7_v34 <= (34'(g6_hi) * 34'(g6_lo) + 34'd32768) >> 16; g7_ip <= g6_ip; g7_z <= g6_z; end
    if (g7_v) dec[g7_h] <= g7_z ? 17'd0 : 17'(g7_v34 >> g7_ip);
  end

  logic          sq_iv, sq_ov;
  logic [SQW-1:0] sq_in [VW], sq_out;
  logic [0:0]    sq_zi [VW];
  logic [1:0]    sq_kind [RL+2];
  int            sq_slot [RL+2];
  logic          sq_v1;
  logic [VW*48-1:0] sq_row;
  logic [1:0]    sq_k0;
  int            sq_s0;

  always_ff @(posedge clk) begin
    if (sq_v1) for (int u = 0; u < VW; u++) sq_in[u] <= SQW'($signed(sq_row[u*48 +: 48]) * $signed(sq_row[u*48 +: 48]));
    sq_kind[0] <= sq_k0; sq_slot[0] <= sq_s0;
    for (int k = 1; k < RL + 2; k++) begin sq_kind[k] <= sq_kind[k-1]; sq_slot[k] <= sq_slot[k-1]; end
  end
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) sq_iv <= 1'b0;
    else sq_iv <= sq_v1;
  always_comb for (int u = 0; u < VW; u++) sq_zi[u] = '0;
  reduce #(.N(VW), .W(SQW), .OP(0), .LPS(2)) u_sq (.clk, .iv(sq_iv), .din(sq_in), .iin(sq_zi), .ov(sq_ov), .dout(sq_out), .iout());

  logic          rf_v0, rf_v1, rf_v2, rf_v3;
  logic [127:0]  rf_m0, rf_m1, rf_m2;
  logic [1:0]    rf_k [5];
  int            rf_s [5];
  logic [3:0]    rf_hi;
  logic [7:0]    rf_hb;
  logic signed [7:0] rf_e2, rf_e3;
  logic [127:0]  rf_mm;
  logic          rf_issue;
  logic [1:0]    rf_ik;
  int            rf_is;

  always_ff @(posedge clk) begin
    rf_k[0] <= rf_ik; rf_s[0] <= rf_is;
    for (int k = 1; k < 5; k++) begin rf_k[k] <= rf_k[k-1]; rf_s[k] <= rf_s[k-1]; end
    if (rf_issue) rf_m0 <= (rf_ik == 2'd2) ? (128'(ssa[2][rf_is]) >> LDV) + 128'(EPS) : 128'(ssa[rf_ik][rf_is]) + 128'(EPS);
    if (rf_v0) begin
      logic [3:0] hi;
      hi = 4'd0;
      for (int k = 1; k < 16; k++) if (rf_m0[k*8 +: 8] != 0) hi = 4'(k);
      rf_hi <= hi; rf_hb <= rf_m0[hi*8 +: 8]; rf_m1 <= rf_m0;
    end
    if (rf_v1) begin
      rf_e2 <= 8'(((rf_hb == 0 ? 0 : int'(rf_hi) * 8 + bitlen8(rf_hb)) - 17) >>> 1);
      rf_m2 <= rf_m1;
    end
    if (rf_v2) begin
      rf_mm <= (rf_e2 >= 0) ? (rf_m2 >> (2 * int'(rf_e2))) : (rf_m2 << (-2 * int'(rf_e2)));
      rf_e3 <= rf_e2;
    end
    if (rf_v3) begin
      fl[rf_k[3]][rf_s[3]] <= t_rsqrt[int'(rf_mm >> 10) - 64];
      fe[rf_k[3]][rf_s[3]] <= rf_e3;
    end
  end
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {rf_v0, rf_v1, rf_v2, rf_v3} <= '0;
    else begin rf_v0 <= rf_issue; rf_v1 <= rf_v0; rf_v2 <= rf_v1; rf_v3 <= rf_v2; end

  logic sc_v1, sc_v2, sc_v3;
  int   sc_s1, sc_r1, sc_s2, sc_r2, sc_s3, sc_r3;
  logic sc_q1, sc_q2, sc_q3;
  logic signed [63:0] sc_p1 [VW];
  v48 sc_y2 [VW];
  logic signed [63:0] sc_p3 [VW];
  logic sc_issue, sc_isq;
  int   sc_is, sc_ir;

  always_ff @(posedge clk) begin
    sc_s1 <= sc_is; sc_r1 <= sc_ir; sc_q1 <= sc_isq;
    sc_s2 <= sc_s1; sc_r2 <= sc_r1; sc_q2 <= sc_q1;
    sc_s3 <= sc_s2; sc_r3 <= sc_r2; sc_q3 <= sc_q2;
    if (sc_issue)
      for (int u = 0; u < VW; u++)
        sc_p1[u] <= 64'(sc_isq ? qr[sc_is][sc_ir * VW + u] : kr[sc_is][sc_ir * VW + u]) * 64'($signed({1'b0, fl[sc_isq ? 0 : 1][sc_is]}));
    if (sc_v1)
      for (int u = 0; u < VW; u++) sc_y2[u] <= rnd(128'(sc_p1[u]), 7 + int'(fe[sc_q1 ? 0 : 1][sc_s1]));
    if (sc_v2)
      for (int u = 0; u < VW; u++)
        if (sc_q2) sc_p3[u] <= 64'(sc_y2[u]) * 64'(LS);
        else kn[sc_s2][sc_r2 * VW + u] <= 19'(sc_y2[u]);
    if (sc_v3 && sc_q3)
      for (int u = 0; u < VW; u++) qn[sc_s3][sc_r3 * VW + u] <= 19'(rnd(128'(sc_p3[u]), 15));
  end
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {sc_v1, sc_v2, sc_v3} <= '0;
    else begin sc_v1 <= sc_issue; sc_v2 <= sc_v1; sc_v3 <= sc_v2; end

  logic [DV*32-1:0] smem [SH][SDEP];
  logic [DV*32-1:0] srd [SH];
  logic [SA-1:0] sra;
  logic p_issue, p1_v, p2_v, p3_v, p4_v;
  int   p_i, p1_i, p2_i, p3_i;
  logic p_two, p1_two, p2_two, p3_two, p4_two;
  logic signed [71:0] p2_pr [SH][DV];
  logic signed [31:0] p2_old [SH][DV];
  logic signed [31:0] p3_nw [SH][DV];
  logic signed [63:0] p4_m [SH][DV];

  always_ff @(posedge clk) begin
    for (int s = 0; s < SH; s++) srd[s] <= smem[s][sra];
    p1_i <= p_i; p2_i <= p1_i; p3_i <= p2_i;
    p1_two <= p_two; p2_two <= p1_two; p3_two <= p2_two; p4_two <= p3_two;
    if (p_issue)
      for (int s = 0; s < SH; s++)
        for (int j = 0; j < DV; j++) begin
          logic signed [31:0] old;
          old = (!p_two && cpos == 0) ? 32'sd0 : $signed(srd[s][j*32 +: 32]);
          p2_old[s][j] <= old;
          p2_pr[s][j] <= p_two ? 72'(kn[s][p_i]) * 72'(dl[s][j]) : 72'(old) * 72'($signed({1'b0, dec[gi * SH + s]}));
        end
    if (p1_v)
      for (int s = 0; s < SH; s++) begin
        logic [DV*32-1:0] w;
        for (int j = 0; j < DV; j++) begin
          logic signed [31:0] nw;
          nw = p1_two ? clip32(72'(p2_old[s][j]) + r16(p2_pr[s][j])) : clip32(r16(p2_pr[s][j]));
          p3_nw[s][j] <= nw;
          w[j*32 +: 32] = nw;
        end
        smem[s][SA'((il * NG + gi) * DK + p1_i)] <= w;
      end
    if (p2_v)
      for (int s = 0; s < SH; s++)
        for (int j = 0; j < DV; j++) p4_m[s][j] <= 64'(p2_two ? qn[s][p2_i] : kn[s][p2_i]) * 64'(p3_nw[s][j]);
    if (p3_v)
      for (int s = 0; s < SH; s++)
        for (int j = 0; j < DV; j++)
          if (p3_two) oa[s][j] <= oa[s][j] + p4_m[s][j];
          else kva[s][j] <= kva[s][j] + p4_m[s][j];
    if (st == D_SC)
      for (int s = 0; s < SH; s++)
        for (int j = 0; j < DV; j++) begin kva[s][j] <= '0; oa[s][j] <= '0; end
  end
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {p1_v, p2_v, p3_v, p4_v} <= '0;
    else begin p1_v <= p_issue; p2_v <= p1_v; p3_v <= p2_v; p4_v <= p3_v; end

  logic d_issue, d1_v;
  int   d_s, d_r, d1_s, d1_r;
  logic signed [49:0] d1_t [VW];
  always_ff @(posedge clk) begin
    d1_s <= d_s; d1_r <= d_r;
    if (d_issue)
      for (int u = 0; u < VW; u++) d1_t[u] <= 50'(vr[d_s][d_r * VW + u]) - 50'(r16(72'(kva[d_s][d_r * VW + u])));
    if (d1_v)
      for (int u = 0; u < VW; u++) dl[d1_s][d1_r * VW + u] <= 50'(r16(72'(d1_t[u]) * 72'($signed({1'b0, beta[gi * SH + d1_s]}))));
  end
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) d1_v <= 1'b0;
    else d1_v <= d_issue;

  logic os_issue, os1_v;
  int   os_s, os_r, os1_s, os1_r;
  always_ff @(posedge clk) begin
    os1_s <= os_s; os1_r <= os_r;
    if (os_issue)
      for (int u = 0; u < VW; u++) ov[os_s][os_r * VW + u] <= 48'(r16(72'(oa[os_s][os_r * VW + u])));
  end
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) os1_v <= 1'b0;
    else os1_v <= os_issue;

  logic ou_issue, ou1_v, ou2_v, ou3_v, ou4_v, ou5_v;
  int   ou_s, ou_r, ou1_s, ou1_r, ou2_s, ou2_r, ou3_s, ou3_r, ou4_s, ou4_r, ou5_s, ou5_r;
  logic signed [63:0] ou1_p [VW];
  logic [16:0] ou1_sg [VW];
  v48 ou1_z [VW];
  v48 ou2_on [VW];
  logic signed [64:0] ou2_zs [VW];
  logic signed [63:0] ou3_g [VW];
  v48 ou3_sz [VW];
  v48 ou4_g [VW], ou4_sz [VW];
  logic signed [95:0] ou5_p [VW];

  always_ff @(posedge clk) begin
    ou1_s <= ou_s; ou1_r <= ou_r; ou2_s <= ou1_s; ou2_r <= ou1_r; ou3_s <= ou2_s; ou3_r <= ou2_r;
    ou4_s <= ou3_s; ou4_r <= ou3_r; ou5_s <= ou4_s; ou5_r <= ou4_r;
    if (ou_issue)
      for (int u = 0; u < VW; u++) begin
        v48 z;
        z = zr[ou_s][ou_r * VW + u];
        ou1_p[u] <= 64'(ov[ou_s][ou_r * VW + u]) * 64'($signed({1'b0, fl[2][ou_s]}));
        ou1_sg[u] <= sfix(z, t_sig[sidx(z)]);
        ou1_z[u] <= z;
      end
    if (ou1_v)
      for (int u = 0; u < VW; u++) begin
        ou2_on[u] <= rnd(128'(ou1_p[u]), 7 + int'(fe[2][ou1_s]));
        ou2_zs[u] <= 65'(ou1_z[u]) * 65'($signed({1'b0, ou1_sg[u]}));
      end
    if (ou2_v)
      for (int u = 0; u < VW; u++) begin
        ou3_g[u] <= 64'(ou2_on[u]) * 64'(t_ng[il * DV + ou2_r * VW + u]);
        ou3_sz[u] <= rnd(128'(ou2_zs[u]), 16);
      end
    if (ou3_v)
      for (int u = 0; u < VW; u++) begin
        ou4_g[u] <= rnd(128'(ou3_g[u]), int'(gf));
        ou4_sz[u] <= ou3_sz[u];
      end
    if (ou4_v)
      for (int u = 0; u < VW; u++) ou5_p[u] <= 96'(ou4_g[u]) * 96'(ou4_sz[u]);
  end
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {ou1_v, ou2_v, ou3_v, ou4_v, ou5_v} <= '0;
    else begin ou1_v <= ou_issue; ou2_v <= ou1_v; ou3_v <= ou2_v; ou4_v <= ou3_v; ou5_v <= ou4_v; end

  logic ld_v;
  int   ld_k;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= D_IDLE; done <= 1'b0; o_we <= 1'b0; cv_issue <= 1'b0; ga_issue <= 1'b0; sq_v1 <= 1'b0; rf_issue <= 1'b0; sc_issue <= 1'b0;
      p_issue <= 1'b0; d_issue <= 1'b0; os_issue <= 1'b0; ou_issue <= 1'b0; ld_v <= 1'b0;
      {c1_v, c2_v, c3_v, c4_v, c5_v} <= '0; {g1_v, g2_v, g3_v, g4_v, g5_v, g6_v, g7_v} <= '0;
      cv_row <= 0; ga_h <= 0;
    end else begin
      done <= 1'b0;
      o_we <= 1'b0;
      cv_issue <= 1'b0;
      ga_issue <= 1'b0;
      sq_v1 <= 1'b0;
      rf_issue <= 1'b0;
      sc_issue <= 1'b0;
      p_issue <= 1'b0;
      d_issue <= 1'b0;
      os_issue <= 1'b0;
      ou_issue <= 1'b0;
      ld_v <= 1'b0;
      c1_v <= cv_issue; c2_v <= c1_v; c3_v <= c2_v; c4_v <= c3_v; c5_v <= c4_v;
      g1_v <= ga_issue; g2_v <= g1_v; g3_v <= g2_v; g4_v <= g3_v; g5_v <= g4_v; g6_v <= g5_v; g7_v <= g6_v;
      if (sq_ov) ssa[sq_kind[RL]][sq_slot[RL]] <= ssa[sq_kind[RL]][sq_slot[RL]] + sq_out;
      case (st)
        D_IDLE: if (go) begin
          il <= int'(imm); base <= fa; dst <= fc; af <= fk[7:0]; gf <= fk[15:8]; cf <= fk[23:16]; cpos <= $signed(pos);
          cmod <= (CK == 4) ? mod3(pos) : 2'(pos % (CK - 1));
          cnt <= 0; st <= D_CONV;
        end
        D_CONV: begin
          if (cnt < CR) begin cv_issue <= 1'b1; cv_row <= cnt; cnt <= cnt + 1; end
          else if (!cv_issue && !c1_v && !c2_v && !c3_v && !c4_v && !c5_v) begin st <= D_WAIT; wcnt <= 0; end
        end
        D_WAIT: begin
          wcnt <= wcnt + 1;
          if (wcnt == 6) begin st <= D_AB; cnt <= 0; end
        end
        D_AB: begin
          if (cnt > 0)
            for (int u = 0; u < VW; u++) begin
              int e;
              e = (cnt - 1) * VW + u - (C + HV * DV) % VW;
              if (e >= 0 && e < HV) breg[e] <= $signed(rd_data[u*48 +: 48]);
              if (e >= HV && e < 2 * HV) areg[e - HV] <= $signed(rd_data[u*48 +: 48]);
            end
          if (cnt == ABR) begin st <= D_GATE; cnt <= 0; end
          else cnt <= cnt + 1;
        end
        D_GATE: begin
          if (cnt < HV) begin ga_issue <= 1'b1; ga_h <= cnt; cnt <= cnt + 1; end
          else if (cnt < HV + 9) cnt <= cnt + 1;
          else begin st <= D_LOAD; gi <= 0; cnt <= 0; wcnt <= 0; end
        end
        D_LOAD: begin
          if (cnt == 0 && !ld_v)
            for (int s = 0; s < SH; s++) for (int k = 0; k < 3; k++) ssa[k][s] <= '0;
          if (ld_v) begin
            int s, part;
            s = ld_k / PR;
            part = ld_k % PR;
            for (int u = 0; u < VW; u++) begin
              v48 x;
              x = $signed(rd_data[u*48 +: 48]);
              if (part < KR) qr[s][part * VW + u] <= x;
              else if (part < 2 * KR) kr[s][(part - KR) * VW + u] <= x;
              else if (part < 2 * KR + VR) vr[s][(part - 2 * KR) * VW + u] <= x;
              else zr[s][(part - 2 * KR - VR) * VW + u] <= x;
            end
            if (part < 2 * KR) begin sq_v1 <= 1'b1; sq_row <= rd_data; sq_k0 <= (part < KR) ? 2'd0 : 2'd1; sq_s0 <= s; end
          end
          if (cnt < SH * PR) begin ld_v <= 1'b1; ld_k <= cnt; cnt <= cnt + 1; end
          else if (!ld_v) begin
            if (wcnt < RL + 6) wcnt <= wcnt + 1;
            else begin st <= D_NF; cnt <= 0; wcnt <= 0; end
          end
        end
        D_NF: begin
          if (cnt < 2 * SH) begin rf_issue <= 1'b1; rf_ik <= 2'(cnt % 2); rf_is <= cnt / 2; cnt <= cnt + 1; end
          else if (cnt < 2 * SH + 6) cnt <= cnt + 1;
          else begin st <= D_SC; cnt <= 0; end
        end
        D_SC: begin
          if (cnt < 2 * SH * KR) begin
            sc_issue <= 1'b1; sc_isq <= (cnt / KR) % 2 == 0; sc_is <= cnt / (2 * KR); sc_ir <= cnt % KR; cnt <= cnt + 1;
          end else if (cnt < 2 * SH * KR + 5) cnt <= cnt + 1;
          else begin st <= D_P1; cnt <= 0; sra <= SA'((il * NG + gi) * DK); end
        end
        D_P1, D_P2: begin
          if (cnt < DK) begin
            p_issue <= 1'b1; p_i <= cnt; p_two <= (st == D_P2); cnt <= cnt + 1;
            sra <= SA'((il * NG + gi) * DK + cnt + 1);
          end else if (cnt < DK + 6) cnt <= cnt + 1;
          else if (st == D_P1) begin st <= D_DL; cnt <= 0; end
          else begin st <= D_OS; cnt <= 0; wcnt <= 0; end
        end
        D_DL: begin
          if (cnt < SH * VR) begin d_issue <= 1'b1; d_s <= cnt / VR; d_r <= cnt % VR; cnt <= cnt + 1; end
          else if (cnt < SH * VR + 3) cnt <= cnt + 1;
          else begin st <= D_P2; cnt <= 0; sra <= SA'((il * NG + gi) * DK); end
        end
        D_OS: begin
          if (cnt < SH * VR) begin os_issue <= 1'b1; os_s <= cnt / VR; os_r <= cnt % VR; cnt <= cnt + 1; end
          if (os1_v) begin
            logic [VW*48-1:0] w;
            for (int u = 0; u < VW; u++) w[u*48 +: 48] = ov[os1_s][os1_r * VW + u];
            sq_v1 <= 1'b1; sq_row <= w; sq_k0 <= 2'd2; sq_s0 <= os1_s;
          end
          if (cnt >= SH * VR && !os_issue && !os1_v) begin
            if (wcnt < RL + 6) wcnt <= wcnt + 1;
            else begin st <= D_OF; cnt <= 0; wcnt <= 0; end
          end
        end
        D_OF: begin
          if (cnt < SH) begin rf_issue <= 1'b1; rf_ik <= 2'd2; rf_is <= cnt; cnt <= cnt + 1; end
          else if (cnt < SH + 6) cnt <= cnt + 1;
          else begin st <= D_OUT; cnt <= 0; end
        end
        D_OUT: begin
          if (cnt < SH * VR) begin ou_issue <= 1'b1; ou_s <= cnt / VR; ou_r <= cnt % VR; cnt <= cnt + 1; end
          else if (cnt < SH * VR + 7) cnt <= cnt + 1;
          else if (gi + 1 < NG) begin gi <= gi + 1; cnt <= 0; wcnt <= 0; st <= D_LOAD; end
          else begin st <= D_DRAIN; wcnt <= 0; end
        end
        D_DRAIN: begin
          wcnt <= wcnt + 1;
          if (wcnt == 6) begin st <= D_IDLE; done <= 1'b1; end
        end
        default: st <= D_IDLE;
      endcase
      if (c5_v) begin
        o_we <= 1'b1;
        o_row <= base / VW + 32'(c5_r);
        for (int u = 0; u < VW; u++) o_data[u] <= rnd(128'(c5_p[u]), 16);
        o_mask <= '1;
        o_first <= (c5_r == 0);
        o_g0 <= 32'(c5_r * VW / 32);
      end
      if (ou5_v) begin
        int h;
        h = gi * SH + ou5_s;
        o_we <= 1'b1;
        o_row <= (dst + 32'(h * DV + ou5_r * VW)) / VW;
        for (int u = 0; u < VW; u++) o_data[u] <= rnd(128'(ou5_p[u]), 16);
        o_mask <= '1;
        o_first <= (h == 0 && ou5_r == 0);
        o_g0 <= 32'((h * DV + ou5_r * VW) / 32);
      end
    end
  end

endmodule
