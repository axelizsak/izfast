module attn #(
  parameter int VK      = 4,
  parameter int HD      = 64,
  parameter int NKV     = 2,
  parameter int REP     = 2,
  parameter int NLAYERS = 4,
  parameter int MAXSEQ  = 256,
  parameter int ATT_TOK = 8
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               tbl_we,
  input  logic [2:0]         tbl_sel,
  input  logic [31:0]        tbl_addr,
  input  logic [16:0]        tbl_data,
  input  logic               tok_start,
  input  logic [31:0]        pos,
  input  logic               q_we,
  input  logic [31:0]        q_row,
  input  logic signed [47:0] q_data [32*VK],
  input  logic [VK-1:0]      q_mask,
  input  logic               go,
  input  logic [31:0]        fc,
  input  logic [31:0]        fk,
  input  logic signed [15:0] imm,
  output logic               done,
  output logic               o_we,
  output logic [31:0]        o_row,
  output logic signed [47:0] o_data [32*VK],
  output logic [VK-1:0]      o_mask,
  output logic               o_first,
  output logic [31:0]        o_g0
);

  localparam int G    = 32;
  localparam int VW   = G * VK;
  localparam int AT   = ATT_TOK;
  localparam int NH   = NKV * REP;
  localparam int NQ   = NH + 2 * NKV;
  localparam int QW   = (VW < HD) ? VW : HD;
  localparam int QR   = HD / QW;
  localparam int SEG  = VW / QW;
  localparam int P    = QW;
  localparam int KVN  = NLAYERS * MAXSEQ * NKV;
  localparam int SR   = MAXSEQ / AT;
  localparam int PSW  = 17 + $clog2(MAXSEQ);
  localparam int L2E  = 47274;
  localparam int LD   = ($clog2(HD) + 1) / 2;
  localparam int LA   = ($clog2(AT) + 1) / 2;
  localparam int LQ   = ($clog2(QW) + 1) / 2;
  localparam int KA   = $clog2(KVN);
  localparam int KB   = KVN / AT;
  localparam int KBA  = (KB > 1) ? $clog2(KB) : 1;

  typedef logic signed [47:0] v48;

  function automatic logic [47:0] abs48(input v48 v);
    abs48 = v[47] ? 48'(-v) : 48'(v);
  endfunction

  function automatic int bitlen(input logic [47:0] v);
    bitlen = 0;
    for (int k = 0; k < 48; k++) if (v[k]) bitlen = k + 1;
  endfunction

  function automatic int sh_a(input logic [47:0] m);
    sh_a = (bitlen(m) > 7) ? bitlen(m) - 7 : 0;
  endfunction

  function automatic int sh_b(input logic [47:0] m, input int s0);
    logic [47:0] w;
    w = (s0 > 0) ? m >> (s0 - 1) : '0;
    sh_b = (s0 > 0 && w[7:0] == 8'hff) ? s0 + 1 : s0;
  endfunction

  function automatic logic signed [7:0] q8(input v48 x, input int s);
    logic signed [48:0] t;
    if (s == 0) t = 49'(x);
    else t = (49'(x) + (49'(1) <<< (s - 1))) >>> s;
    q8 = (t > 127) ? 8'sd127 : (t < -127) ? -8'sd127 : t[7:0];
  endfunction

  function automatic v48 rnd(input logic signed [127:0] v, input int n);
    logic signed [127:0] t;
    if (n <= 0) rnd = 48'(v <<< (-n));
    else if (n >= 120) rnd = '0;
    else begin
      t = v + (128'(1) <<< (n - 1));
      rnd = 48'(t >>> n);
    end
  endfunction

  function automatic int slot(input int l, input int t, input int g);
    slot = (l * MAXSEQ + t) * NKV + g;
  endfunction

  function automatic int brow(input int l, input int tb, input int g);
    brow = (l * SR + tb) * NKV + g;
  endfunction

  logic [QW*48-1:0]  qbuf [NQ*QR];
  logic [5:0]        kminp [KVN], vminp [KVN];
  logic [AT*64-1:0]  scm [2*REP*SR];
  logic [16:0]       t_exp2 [256];
  logic [HD*8-1:0]   t_cos [MAXSEQ];
  logic [HD*8-1:0]   t_sin [MAXSEQ];

  always_ff @(posedge clk) begin
    if (tbl_we && tbl_sel == 3'd1) t_exp2[tbl_addr] <= tbl_data;
    if (tbl_we && tbl_sel == 3'd3) t_cos[tbl_addr / (HD / 2)][(tbl_addr % (HD / 2)) * 16 +: 16] <= tbl_data[15:0];
    if (tbl_we && tbl_sel == 3'd4) t_sin[tbl_addr / (HD / 2)][(tbl_addr % (HD / 2)) * 16 +: 16] <= tbl_data[15:0];
  end

  logic [HD*8-1:0] crow, srow;
  always_ff @(posedge clk) if (tok_start) begin crow <= t_cos[pos]; srow <= t_sin[pos]; end

  logic        k1_v, k2_v, k3_v;
  logic [31:0] k1_row, k2_row, k3_row;
  logic [VK-1:0] k1_m, k2_m, k3_m;
  logic signed [63:0] k1_ac [VW/2], k1_bs [VW/2], k1_as [VW/2], k1_bc [VW/2];
  v48          k1_raw [VW];
  logic        k1_rot [VW];
  v48          k2_x [VW];
  logic [47:0] k3_a [VW];
  logic [47:0] mx_in [SEG][QW];
  logic [47:0] mx_seg [SEG];
  logic        mx_ov;
  logic [LQ:0] kq_dl;
  logic [31:0] kq_row [LQ+1];
  logic [VK-1:0] kq_m [LQ+1];
  logic [47:0] mxh [NQ];

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {k1_v, k2_v, k3_v} <= '0;
    else begin k1_v <= q_we; k2_v <= k1_v; k3_v <= k2_v; end

  always_ff @(posedge clk) begin
    if (q_we) begin
      k1_row <= q_row; k1_m <= q_mask;
      for (int p = 0; p < VW / 2; p++) begin
        int e, k;
        logic signed [15:0] c, s;
        e = int'(q_row) * VW + 2 * p;
        k = (e % HD) / 2;
        c = crow[k*16 +: 16];
        s = srow[k*16 +: 16];
        k1_ac[p] <= q_data[2*p] * c;
        k1_bs[p] <= q_data[2*p+1] * s;
        k1_as[p] <= q_data[2*p] * s;
        k1_bc[p] <= q_data[2*p+1] * c;
      end
      for (int u = 0; u < VW; u++) begin
        k1_raw[u] <= q_data[u];
        k1_rot[u] <= (int'(q_row) * VW + u) / HD < NH + NKV;
      end
    end
    if (k1_v) begin
      k2_row <= k1_row; k2_m <= k1_m;
      for (int p = 0; p < VW / 2; p++) begin
        k2_x[2*p]   <= k1_rot[2*p] ? rnd(128'(k1_ac[p] - k1_bs[p]), 14) : k1_raw[2*p];
        k2_x[2*p+1] <= k1_rot[2*p] ? rnd(128'(k1_as[p] + k1_bc[p]), 14) : k1_raw[2*p+1];
      end
    end
    if (k2_v) begin
      k3_row <= k2_row; k3_m <= k2_m;
      for (int s = 0; s < SEG; s++)
        if (k2_m[(s * QW) / G]) begin
          logic [QW*48-1:0] w;
          for (int u = 0; u < QW; u++) w[u*48 +: 48] = k2_x[s*QW+u];
          qbuf[(int'(k2_row) * VW) / QW + s] <= w;
        end
      for (int u = 0; u < VW; u++) k3_a[u] <= abs48(k2_x[u]);
    end
  end

  always_comb for (int s = 0; s < SEG; s++) for (int u = 0; u < QW; u++) mx_in[s][u] = k3_a[s*QW+u];
  mreduce #(.M(SEG), .N(QW), .W(48), .OP(1)) u_mx (.clk, .iv(k3_v), .din(mx_in), .ov(mx_ov), .dout(mx_seg));

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) kq_dl <= '0;
    else kq_dl <= {kq_dl[LQ-1:0], k3_v};
  always_ff @(posedge clk) begin
    kq_row[0] <= k3_row; kq_m[0] <= k3_m;
    for (int k = 1; k <= LQ; k++) begin kq_row[k] <= kq_row[k-1]; kq_m[k] <= kq_m[k-1]; end
  end
  logic [31:0] mrow;
  logic [VK-1:0] mmask;
  assign mrow  = (LQ == 0) ? k3_row : kq_row[LQ-1];
  assign mmask = (LQ == 0) ? k3_m : kq_m[LQ-1];
  always_ff @(posedge clk)
    if (mx_ov)
      for (int s = 0; s < SEG; s++)
        if (mmask[(s * QW) / G]) begin
          int qr, h;
          qr = (int'(mrow) * VW) / QW + s;
          h = qr / QR;
          if (qr % QR == 0 || mx_seg[s] > mxh[h]) mxh[h] <= mx_seg[s];
        end
  logic rx_busy;
  assign rx_busy = q_we | k1_v | k2_v | k3_v | (|kq_dl);

  typedef enum logic [2:0] {F_IDLE, F_META, F_SHA, F_SHB, F_QUANT, F_SCORE, F_DONE} fstate_t;
  fstate_t fst;
  logic        active;
  int          lay, ntok, fcnt, fr, ft;
  logic [30:0] lf;
  logic [31:0] obase;
  int          s0q [REP], s0k, s0v, kqs, vqs, kpd, vpd;
  int          qshv [2][REP], kminb [2], vminb [2];
  logic signed [63:0] amax [2][REP];
  logic [HD*8-1:0] qq [REP];
  logic [5:0]  kminp_q, vminp_q;
  logic [QW*48-1:0] qrd [REP], krd, vrd;
  logic        fq_v;
  int          fq_r;
  logic        sh_we, kv_we;
  int          shb_kq, shb_vq, shb_km, shb_vm;
  logic [HD*8-1:0] kst, vst;

  assign sh_we = active && fst == F_SHB;
  always_comb begin
    shb_kq = sh_b(mxh[NH + fcnt], s0k);
    shb_vq = sh_b(mxh[NH + NKV + fcnt], s0v);
    shb_km = (ntok == 1 || kpd > shb_kq) ? shb_kq : kpd;
    shb_vm = (ntok == 1 || vpd > shb_vq) ? shb_vq : vpd;
  end

  always_ff @(posedge clk) begin
    if (fst == F_META) begin
      kminp_q <= kminp[KA'(slot(lay, int'(pos) - 1, fcnt))];
      vminp_q <= vminp[KA'(slot(lay, int'(pos) - 1, fcnt))];
    end
    if (sh_we) begin
      kminp[KA'(slot(lay, ntok - 1, fcnt))] <= 6'(shb_km);
      vminp[KA'(slot(lay, ntok - 1, fcnt))] <= 6'(shb_vm);
    end
    if (fst == F_QUANT && fr < QR) begin
      for (int j = 0; j < REP; j++) qrd[j] <= qbuf[(fcnt * REP + j) * QR + fr];
      krd <= qbuf[(NH + fcnt) * QR + fr];
      vrd <= qbuf[(NH + NKV + fcnt) * QR + fr];
    end
  end

  logic        t1_v;
  int          t1_t;
  logic        fscore_issue;
  assign fscore_issue = active && (fst == F_SCORE) && (ft < ntok);
  logic [HD*8-1:0] krow [AT];
  logic [5:0]  kshr [AT];


  logic [23:0] prod_d [REP*AT][HD];
  logic [23:0] dot [REP*AT];
  logic        dot_ov;
  logic        t2_v;
  int          t2_t;
  logic [5:0]  t2_ksh [AT];
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) t2_v <= 1'b0;
    else t2_v <= t1_v;
  always_ff @(posedge clk)
    if (t1_v) begin
      t2_t <= t1_t;
      t2_ksh <= kshr;
      for (int j = 0; j < REP; j++)
        for (int u = 0; u < AT; u++)
          for (int e = 0; e < HD; e++)
            prod_d[j*AT+u][e] <= 24'($signed(qq[j][e*8 +: 8]) * $signed(krow[u][e*8 +: 8]));
    end
  mreduce #(.M(REP*AT), .N(HD), .W(24), .OP(0)) u_dot (.clk, .iv(t2_v), .din(prod_d), .ov(dot_ov), .dout(dot));

  logic [LD:0] td_v;
  int          td_t [LD+1];
  logic [5:0]  td_ksh [LD+1][AT];
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) td_v <= '0;
    else td_v <= {td_v[LD-1:0], t2_v};
  always_ff @(posedge clk) begin
    td_t[0] <= t2_t; td_ksh[0] <= t2_ksh;
    for (int k = 1; k <= LD; k++) begin td_t[k] <= td_t[k-1]; td_ksh[k] <= td_ksh[k-1]; end
  end

  logic        ta_v;
  int          ta_t;
  logic [63:0] ta_a [REP][AT];
  logic        am_ov;
  logic [63:0] cmax [REP];
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) ta_v <= 1'b0;
    else ta_v <= dot_ov;
  always_ff @(posedge clk)
    if (dot_ov) begin
      int t, fb;
      t = td_t[LD-1];
      fb = fcnt & 1;
      ta_t <= t;
      for (int j = 0; j < REP; j++) begin
        logic [AT*64-1:0] row;
        for (int u = 0; u < AT; u++) begin
          logic signed [63:0] a;
          a = 64'($signed(dot[j*AT+u])) <<< (int'(td_ksh[LD-1][u]) - kminb[fb]);
          ta_a[j][u] <= (t + u < ntok) ? a : {1'b1, 63'd0};
          row[u*64 +: 64] = a;
        end
        scm[(fb * REP + j) * SR + t / AT] <= row;
      end
    end
  mreduce #(.M(REP), .N(AT), .W(64), .OP(2)) u_am (.clk, .iv(ta_v), .din(ta_a), .ov(am_ov), .dout(cmax));
  logic [LA:0] tm_dl;
  int          tm_t [LA+1];
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) tm_dl <= '0;
    else tm_dl <= {tm_dl[LA-1:0], ta_v};
  always_ff @(posedge clk) begin
    tm_t[0] <= ta_t;
    for (int k = 1; k <= LA; k++) tm_t[k] <= tm_t[k-1];
  end
  logic [31:0] am_t;
  assign am_t = (LA == 0) ? 32'(ta_t) : 32'(tm_t[LA-1]);
  logic f_inflight;
  assign f_inflight = t1_v | t2_v | (|td_v) | ta_v | (|tm_dl);

  typedef enum logic [1:0] {O_IDLE, O_DIV, O_OUT} ostate_t;
  ostate_t ost;
  logic        prun;
  int          pcnt, ocnt, bt, dbit, oj, orr, ob;
  int          cvmin [2];
  logic [PSW-1:0] psum [2][REP];
  logic [PSW:0] rem [REP];
  logic [16:0] quo [REP];
  logic signed [95:0] oacc [2][REP][HD];

  logic [AT*64-1:0] scr [REP];
  logic        p1_v, p2_v, p3_v, p4_v, p5_v, p6_v;
  int          p1_t, p2_t, p3_t, p4_t, p5_t, p6_t;
  logic [63:0] p2_z [REP][AT];
  logic [79:0] p3_lo [REP][AT];
  logic [78:0] p3_hi [REP][AT];
  logic [95:0] p4_zl [REP][AT];
  logic        p5_zero [REP][AT];
  logic [4:0]  p5_ip [REP][AT];
  logic [16:0] e2q [REP][AT];
  logic [16:0] p6_p [REP][AT];
  logic [HD*8-1:0] vrow [AT];
  logic [5:0]  vshr [AT];
  logic        b_issue;
  int          bb;
  assign bb = pcnt & 1;
  assign b_issue = active && prun && (bt < ntok);

  always_ff @(posedge clk) begin
    if (b_issue) for (int j = 0; j < REP; j++) scr[j] <= scm[(bb * REP + j) * SR + bt / AT];
  end

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) kv_we <= 1'b0;
    else kv_we <= fq_v && fq_r == QR - 1;

  always_ff @(posedge clk)
    if (fq_v) begin
      logic [QW*8-1:0] kp, vp, qp;
      for (int u = 0; u < QW; u++) begin
        kp[u*8 +: 8] = q8(v48'(krd[u*48 +: 48]), kqs);
        vp[u*8 +: 8] = q8(v48'(vrd[u*48 +: 48]), vqs);
      end
      kst <= (HD*8)'({kp, kst} >> (QW*8));
      vst <= (HD*8)'({vp, vst} >> (QW*8));
      for (int j = 0; j < REP; j++) begin
        for (int u = 0; u < QW; u++) qp[u*8 +: 8] = q8(v48'(qrd[j][u*48 +: 48]), qshv[fcnt & 1][j]);
        qq[j] <= (HD*8)'({qp, qq[j]} >> (QW*8));
      end
    end

  for (genvar u = 0; u < AT; u++) begin : g_kv
    logic [HD*8-1:0] kmem [KB];
    logic [HD*8-1:0] vmem [KB];
    logic [5:0]      kshm [KB];
    logic [5:0]      vshm [KB];
    logic [HD*8-1:0] kr, vr;
    logic [5:0]      ksr, vsr;
    always_ff @(posedge clk) begin
      if (kv_we && (ntok - 1) % AT == u) begin
        kmem[KBA'(brow(lay, (ntok - 1) / AT, fcnt))] <= kst;
        vmem[KBA'(brow(lay, (ntok - 1) / AT, fcnt))] <= vst;
      end
      if (sh_we && (ntok - 1) % AT == u) begin
        kshm[KBA'(brow(lay, (ntok - 1) / AT, fcnt))] <= 6'(shb_kq);
        vshm[KBA'(brow(lay, (ntok - 1) / AT, fcnt))] <= 6'(shb_vq);
      end
      if (fscore_issue) begin
        kr <= kmem[KBA'(brow(lay, ft / AT, fcnt))];
        ksr <= kshm[KBA'(brow(lay, ft / AT, fcnt))];
      end
      if (p5_v) begin
        vr <= vmem[KBA'(brow(lay, p5_t / AT, pcnt))];
        vsr <= vshm[KBA'(brow(lay, p5_t / AT, pcnt))];
      end
    end
    assign krow[u] = kr;
    assign kshr[u] = ksr;
    assign vrow[u] = vr;
    assign vshr[u] = vsr;
  end

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {p1_v, p2_v, p3_v, p4_v, p5_v, p6_v} <= '0;
    else begin
      p1_v <= b_issue;
      p2_v <= p1_v;
      p3_v <= p2_v;
      p4_v <= p3_v;
      p5_v <= p4_v;
      p6_v <= p5_v;
    end

  always_ff @(posedge clk) begin
    if (b_issue) p1_t <= bt;
    if (p1_v) begin
      p2_t <= p1_t;
      for (int j = 0; j < REP; j++)
        for (int u = 0; u < AT; u++) p2_z[j][u] <= 64'(amax[bb][j] - $signed(scr[j][u*64 +: 64]));
    end
    if (p2_v) begin
      p3_t <= p2_t;
      for (int j = 0; j < REP; j++)
        for (int u = 0; u < AT; u++) begin
          p3_lo[j][u] <= p2_z[j][u] * lf[15:0];
          p3_hi[j][u] <= p2_z[j][u] * lf[30:16];
        end
    end
    if (p3_v) begin
      p4_t <= p3_t;
      for (int j = 0; j < REP; j++)
        for (int u = 0; u < AT; u++) p4_zl[j][u] <= 96'(p3_lo[j][u]) + (96'(p3_hi[j][u]) << 16);
    end
    if (p4_v) begin
      p5_t <= p4_t;
      for (int j = 0; j < REP; j++) begin
        int n;
        n = 54 - qshv[bb][j] - kminb[bb];
        for (int u = 0; u < AT; u++) begin
          logic [95:0] zl, uu;
          logic big;
          zl = p4_zl[j][u];
          if (n <= 0) begin
            logic [127:0] w;
            w = 128'(zl) << (-n);
            big = (-n >= 32) ? (zl != 0) : (w >= 128'(17 * 256));
            uu = 96'(w);
          end
          else begin
            logic [96:0] t;
            t = {1'b0, zl} + (97'(1) << (n - 1));
            uu = 96'(t >> n);
            big = uu >= 96'(17 * 256);
          end
          p5_zero[j][u] <= big || (p4_t + u >= ntok);
          p5_ip[j][u] <= 5'(uu >> 8);
          e2q[j][u] <= t_exp2[uu[7:0]];
        end
      end
    end
    if (p5_v) begin
      p6_t <= p5_t;
      for (int j = 0; j < REP; j++)
        for (int u = 0; u < AT; u++) p6_p[j][u] <= p5_zero[j][u] ? 17'd0 : (e2q[j][u] >> p5_ip[j][u]);
    end
  end

  logic [95:0] sv [REP*HD][AT];
  logic [95:0] vsum [REP*HD];
  logic        vs_ov;
  logic [PSW-1:0] pin [REP][AT];
  logic [PSW-1:0] psc [REP];
  logic        ps_ov;
  logic        p7_v;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) p7_v <= 1'b0;
    else p7_v <= p6_v;
  always_ff @(posedge clk)
    if (p6_v)
      for (int j = 0; j < REP; j++)
        for (int u = 0; u < AT; u++) begin
          pin[j][u] <= PSW'(p6_p[j][u]);
          for (int e = 0; e < HD; e++)
            sv[j*HD+e][u] <= 96'($signed({1'b0, p6_p[j][u]}) * $signed(vrow[u][e*8 +: 8])) <<< (int'(vshr[u]) - vminb[bb]);
        end
  mreduce #(.M(REP), .N(AT), .W(PSW), .OP(0)) u_ps (.clk, .iv(p7_v), .din(pin), .ov(ps_ov), .dout(psc));
  mreduce #(.M(REP*HD), .N(AT), .W(96), .OP(0)) u_vs (.clk, .iv(p7_v), .din(sv), .ov(vs_ov), .dout(vsum));
  logic [LA:0] vd_dl;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) vd_dl <= '0;
    else vd_dl <= {vd_dl[LA-1:0], p7_v};
  logic b_inflight;
  assign b_inflight = b_issue | p1_v | p2_v | p3_v | p4_v | p5_v | p6_v | p7_v | (|vd_dl);
  logic acc_v;
  assign acc_v = (LA == 0) ? p7_v : vd_dl[LA-1];

  logic        o1_v, o2_v;
  logic [31:0] o1_row;
  int          o1_off, o1_j;
  logic        o1_first;
  logic [31:0] o1_g0;
  int          o1_vmin;
  logic signed [112:0] o1_p [P];
  logic signed [95:0] ochunk [P];
  v48          odv [P];
  logic        oclr;

  assign oclr = active && ost == O_OUT && oj >= REP && !o1_v;
  always_comb
    for (int u = 0; u < P; u++) begin
      ochunk[u] = '0;
      odv[u] = rnd(128'(o1_p[u]), 32 - o1_vmin);
      for (int b = 0; b < 2; b++)
        for (int j = 0; j < REP; j++)
          for (int r = 0; r < HD / P; r++)
            if (ob == b && oj == j && orr == r) ochunk[u] = oacc[b][j][r * P + u];
    end

  always_ff @(posedge clk)
    for (int b = 0; b < 2; b++)
      for (int j = 0; j < REP; j++)
        if (go || (oclr && ob == b)) begin
          psum[b][j] <= '0;
          for (int e = 0; e < HD; e++) oacc[b][j][e] <= '0;
        end else if (acc_v && bb == b) begin
          psum[b][j] <= psum[b][j] + psc[j];
          for (int e = 0; e < HD; e++) oacc[b][j][e] <= oacc[b][j][e] + $signed(vsum[j*HD+e]);
        end

  always_ff @(posedge clk)
    if (o1_v) begin
      o_row <= o1_row;
      o_first <= o1_first;
      o_g0 <= o1_g0;
      for (int k = 0; k < VK; k++) o_mask[k] <= (k * G >= o1_off) && (k * G < o1_off + P);
      for (int k = 0; k < VW / P; k++)
        if (o1_off == k * P)
          for (int u = 0; u < P; u++) o_data[k * P + u] <= odv[u];
    end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fst <= F_IDLE; ost <= O_IDLE; prun <= 1'b0; active <= 1'b0; done <= 1'b0; fcnt <= 0; pcnt <= 0; ocnt <= 0;
      t1_v <= 1'b0; fq_v <= 1'b0; o1_v <= 1'b0; o_we <= 1'b0;
    end else begin
      done <= 1'b0;
      t1_v <= 1'b0;
      fq_v <= 1'b0;
      o1_v <= 1'b0;
      o_we <= 1'b0;
      if (go) begin
        active <= 1'b1;
        lay <= int'(imm);
        ntok <= int'(pos) + 1;
        lf <= 31'(L2E * fk);
        obase <= fc;
        fcnt <= 0;
        pcnt <= 0;
        ocnt <= 0;
        prun <= 1'b0;
        fst <= F_IDLE;
        ost <= O_IDLE;
      end

      if (active) case (fst)
        F_IDLE: if (!rx_busy && fcnt < NKV && fcnt <= pcnt + 1) fst <= F_META;
        F_META: fst <= F_SHA;
        F_SHA: begin
          for (int j = 0; j < REP; j++) s0q[j] <= sh_a(mxh[fcnt * REP + j]);
          s0k <= sh_a(mxh[NH + fcnt]);
          s0v <= sh_a(mxh[NH + NKV + fcnt]);
          kpd <= int'(kminp_q);
          vpd <= int'(vminp_q);
          fst <= F_SHB;
        end
        F_SHB: begin
          int fb;
          fb = fcnt & 1;
          for (int j = 0; j < REP; j++) qshv[fb][j] <= sh_b(mxh[fcnt * REP + j], s0q[j]);
          kqs <= shb_kq; vqs <= shb_vq;
          kminb[fb] <= shb_km; vminb[fb] <= shb_vm;
          fr <= 0;
          fst <= F_QUANT;
        end
        F_QUANT: begin
          if (fr < QR) begin fq_v <= 1'b1; fq_r <= fr; fr <= fr + 1; end
          else if (!fq_v) begin ft <= 0; fst <= F_SCORE; end
        end
        F_SCORE: begin
          if (ft < ntok) begin t1_v <= 1'b1; t1_t <= ft; ft <= ft + AT; end
          else if (!f_inflight) begin fcnt <= fcnt + 1; fst <= F_IDLE; end
        end
        default: fst <= F_IDLE;
      endcase

      for (int j = 0; j < REP; j++)
        if (am_ov) begin
          int fb;
          fb = fcnt & 1;
          if (am_t == 0 || $signed(cmax[j]) > amax[fb][j]) amax[fb][j] <= $signed(cmax[j]);
        end

      if (active && !prun && pcnt < fcnt && pcnt <= ocnt + 1) begin prun <= 1'b1; bt <= 0; end
      if (prun) begin
        if (bt < ntok) bt <= bt + AT;
        else if (!b_inflight) begin
          prun <= 1'b0;
          cvmin[bb] <= vminb[bb];
          pcnt <= pcnt + 1;
        end
      end

      if (active) case (ost)
        O_IDLE: if (ocnt < pcnt) begin
          ob <= ocnt & 1;
          dbit <= 16;
          for (int j = 0; j < REP; j++) begin rem[j] <= (PSW+1)'(1 << 15); quo[j] <= '0; end
          ost <= O_DIV;
        end
        O_DIV: begin
          for (int j = 0; j < REP; j++) begin
            logic [PSW:0] r;
            logic [16:0] q;
            r = rem[j]; q = quo[j];
            for (int k = 0; k < 4; k++)
              if (dbit - k >= 0) begin
                if ({r[PSW-1:0], 1'b0} >= {1'b0, psum[ob][j]}) begin r = {r[PSW-1:0], 1'b0} - {1'b0, psum[ob][j]}; q[dbit - k] = 1'b1; end
                else r = {r[PSW-1:0], 1'b0};
              end
            rem[j] <= r; quo[j] <= q;
          end
          dbit <= dbit - 4;
          if (dbit < 4) begin oj <= 0; orr <= 0; ost <= O_OUT; end
        end
        O_OUT: begin
          if (oj < REP) begin
            int idx0;
            idx0 = (ocnt * REP + oj) * HD + orr * P;
            o1_v <= 1'b1;
            o1_row <= obase / VW + 32'(idx0 / VW);
            o1_off <= idx0 % VW;
            o1_j <= oj;
            o1_first <= (ocnt == 0 && oj == 0 && orr == 0);
            o1_g0 <= 32'((idx0 - idx0 % VW) / G);
            o1_vmin <= cvmin[ob];
            for (int u = 0; u < P; u++) o1_p[u] <= 113'(ochunk[u] * $signed({1'b0, quo[oj]}));
            if (orr + 1 < HD / P) orr <= orr + 1;
            else begin orr <= 0; oj <= oj + 1; end
          end else if (!o1_v) begin
            ocnt <= ocnt + 1;
            if (ocnt + 1 >= NKV) begin active <= 1'b0; done <= 1'b1; end
            ost <= O_IDLE;
          end
        end
        default: ost <= O_IDLE;
      endcase

      if (o1_v) o_we <= 1'b1;
    end
  end

endmodule
