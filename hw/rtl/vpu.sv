module vpu #(
  parameter int VK     = 4,
  parameter int WBITS  = 4,
  parameter int VDEPTH = 4096,
  parameter int EDEPTH = 1024,
  parameter int XDEPTH = 32
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   emb_we,
  input  logic [31:0]            emb_addr,
  input  logic [32*WBITS+15:0]   emb_data,
  input  logic                   tbl_we,
  input  logic [2:0]             tbl_sel,
  input  logic [31:0]            tbl_addr,
  input  logic [16:0]            tbl_data,
  input  logic [31:0]            vm_raddr,
  output logic signed [47:0]     vm_rdata,
  input  logic                   go,
  input  logic [7:0]             op,
  input  logic [31:0]            fa, fb, fc, fn, fm, fk,
  input  logic signed [15:0]     imm,
  input  logic [7:0]             flags,
  input  logic [31:0]            tok,
  output logic                   done,
  output logic                   idle,
  output logic [31:0]            tok_out,
  output logic                   x_we,
  output logic [VK-1:0]          x_mask,
  output logic [31:0]            x_addr,
  output logic [32*8+4:0]        x_data [VK],
  output logic                   mv_start,
  output logic [31:0]            mv_base, mv_rows, mv_groups,
  input  logic                   mv_done,
  input  logic [31:0]            y_ready,
  output logic [31:0]            y_addr,
  output logic                   y_re,
  input  logic signed [63:0]     y_vec [2*32*VK],
  output logic                   q_we,
  output logic [31:0]            q_row,
  output logic signed [47:0]     q_data [32*VK],
  output logic [VK-1:0]          q_mask,
  input  logic                   o_we,
  input  logic [31:0]            o_row,
  input  logic signed [47:0]     o_data [32*VK],
  input  logic [VK-1:0]          o_mask,
  input  logic                   o_first,
  input  logic [31:0]            o_g0
);

  localparam int G     = 32;
  localparam int VW    = G * VK;
  localparam int WW    = G * WBITS + 16;
  localparam int XW    = G * 8 + 5;
  localparam int VROWS = VDEPTH / VW;
  localparam int VA    = $clog2(VROWS);
  localparam int EA    = $clog2(EDEPTH);
  localparam int XA    = $clog2(XDEPTH);
  localparam int DMAX  = 24;
  localparam int EPS   = 43;
  localparam int SQW   = 96 + $clog2(VW) + 1;
  localparam int SSW   = 112;
  localparam int GL    = 3;
  localparam int SL    = ($clog2(VW) + 1) / 2;
  localparam logic [7:0] OP_EMBED = 1, OP_QMV = 2, OP_RESCALE = 3, OP_ARGMAX = 5;

  typedef logic signed [47:0] v48;

  function automatic logic [47:0] abs48(input v48 v);
    abs48 = v[47] ? 48'(-v) : 48'(v);
  endfunction

  function automatic int bitlen(input logic [127:0] v);
    bitlen = 0;
    for (int k = 0; k < 128; k++) if (v[k]) bitlen = k + 1;
  endfunction

  function automatic int sh_a(input logic [47:0] m);
    int b;
    b = bitlen({80'd0, m});
    sh_a = (b > 7) ? b - 7 : 0;
  endfunction

  function automatic int sh_b(input logic [47:0] m, input int s0);
    logic [47:0] w;
    w = (s0 > 0) ? m >> (s0 - 1) : '0;
    sh_b = (s0 > 0 && w[7:0] == 8'hff) ? s0 + 1 : s0;
  endfunction

  function automatic v48 rnd48(input logic signed [95:0] v, input int n);
    logic signed [96:0] t;
    if (n <= 0) rnd48 = 48'(v <<< (-n));
    else if (n >= 96) rnd48 = '0;
    else begin
      t = 97'(v) + (97'(1) <<< (n - 1));
      rnd48 = 48'(t >>> n);
    end
  endfunction

  function automatic logic signed [7:0] q8(input v48 x, input int s);
    logic signed [48:0] t;
    if (s == 0) t = 49'(x);
    else t = (49'(x) + (49'(1) <<< (s - 1))) >>> s;
    q8 = (t > 127) ? 8'sd127 : (t < -127) ? -8'sd127 : t[7:0];
  endfunction

  logic [VW*48-1:0] vmem [VROWS];
  logic [WW-1:0]    emb  [EDEPTH];
  logic [15:0]      t_rsqrt [192];
  logic [16:0]      t_sig   [1024];
  logic [5:0]       gsh  [XDEPTH];

`ifndef SYNTHESIS
  initial begin
    string f;
    if ($value$plusargs("emb=%s", f)) $readmemh(f, emb);
  end
`endif

  always_ff @(posedge clk) begin
    if (emb_we) emb[EA'(emb_addr)] <= emb_data;
    if (tbl_we && tbl_sel == 3'd0) t_rsqrt[tbl_addr] <= tbl_data[15:0];
    if (tbl_we && tbl_sel == 3'd2) t_sig[tbl_addr] <= tbl_data;
  end
  assign vm_rdata = vmem[VA'(vm_raddr / VW)][(vm_raddr % VW) * 48 +: 48];

  typedef enum logic [3:0] {V_IDLE, V_EMB, V_QSET, V_QMV, V_RSET, V_RES, V_ARG, V_DRAIN} vstate_t;
  vstate_t st;
  logic [7:0]  cop, cfl;
  logic [31:0] ca, cb, cc, cn, cm, ck, ctok;
  logic signed [15:0] cimm;
  logic [31:0] i;
  logic        mv_pend, rms_ok;
  int          ref_sh, s_const;
  int          shmin, shmax;
  logic [5:0]  xsh;
  logic [15:0] rms_l;
  logic signed [7:0] rms_e;
  logic [SSW-1:0] ss;

  logic        wi_v, wi_trk, wi_ss, wi_first;
  logic [31:0] wi_row, wi_g0;
  v48          wi_d [VW];
  logic [VK-1:0] wi_m;

  logic        e1_v;
  logic [31:0] e1_i;
  logic [WW-1:0] ewd [VK];

  logic        q1_v;
  logic [31:0] q1_i;
  logic [VW*48-1:0] vrd;
  logic [31:0] vra;

  logic        r1_v, r2_v, r3_v, r4_v, r5_v, r6_v, r7_v;
  logic [31:0] r1_i, r2_i, r3_i, r4_i, r5_i, r6_i, r7_i;
  logic signed [79:0] r2_y [2*VW];
  v48          r3_x [2*VW];
  v48          r4_g [VW], r4_u [VW], r5_u [VW], r6_u [VW];
  logic        r4_hi [VW], r4_lo [VW];
  logic [16:0] sigq [VW];
  logic signed [65:0] r5_gs [VW];
  logic signed [49:0] r6_s [VW];
  logic signed [65:0] r7_p0 [VW], r7_p1 [VW], r7_p2 [VW];

  logic        a1_v;
  logic [31:0] a1_i;
  logic [63:0] a_d [VW];
  logic [31:0] a_ix [VW];
  logic        a_ov;
  logic [63:0] a_max;
  logic [31:0] a_imax;
  logic signed [63:0] best;
  logic [31:0] besti;
  logic        a_first;

  logic        rms_go, rms_busy;

  logic        w1_v, w2_v, w3_v;
  logic [47:0] w1_a [VW];
  logic [VK-1:0] w1_m, w2_m, w3_m;
  logic        w1_trk, w1_ss, w1_first, w2_trk, w2_ss, w2_first, w3_ss, w3_first;
  logic [31:0] w1_g0;
  logic [31:0] w2_p [VW][6];
  logic [SQW-1:0] w3_sq [VW];
  logic        md_v [GL+1];
  logic [VK-1:0] md_m [GL+1];
  logic        md_trk [GL+1], md_first [GL+1];
  logic [31:0] md_g0 [GL+1];
  logic        sd_v [SL+1], sd_first [SL+1], sd_ss [SL+1];
  logic [47:0] gm [VK];
  logic        gm_v;
  logic [47:0] gin [VK][G];
  logic        w5_v, w5_trk, w5_first;
  logic [VK-1:0] w5_m;
  logic [31:0] w5_g0;
  logic [47:0] w5_gm [VK];
  int          w5_s0 [VK];
  logic [SQW-1:0] ss_chunk;
  logic        ss_ov;
  logic [0:0]  zi [VW];

  logic wr_busy;
  always_comb begin
    wr_busy = wi_v | w1_v | w2_v | w3_v | w5_v;
    for (int k = 0; k <= GL; k++) wr_busy |= md_v[k];
    for (int k = 0; k <= SL; k++) wr_busy |= sd_v[k];
  end
  logic a_busy, pipe_busy;
  assign pipe_busy = e1_v | q1_v | r1_v | r2_v | r3_v | r4_v | r5_v | r6_v | r7_v | a1_v | a_busy | rms_busy;
  assign idle = (st == V_IDLE) && !wr_busy && !pipe_busy;
  assign tok_out = besti;

  always_comb begin
    vra = 32'(cc / VW) + r2_i / VW;
    if (st == V_QMV) vra = 32'(ca / VW) + i / VW;
  end
  always_ff @(posedge clk) if (st == V_QMV || r2_v) vrd <= vmem[VA'(vra)];

  logic [31:0] ewd_a [VK];
  always_comb
    for (int k = 0; k < VK; k++) ewd_a[k] = ctok * (cn >> 5) + (i >> 5) + 32'(k);
  always_ff @(posedge clk)
    if (st == V_EMB) for (int k = 0; k < VK; k++) ewd[k] <= emb[EA'(ewd_a[k])];

  logic rs_ready;
  always_comb begin
    int need, rows_n;
    rows_n = cfl[2] ? 2 * int'(cn) : int'(cn);
    need = cfl[2] ? 2 * (int'(i) + VW) : int'(i) + VW;
    if (need > rows_n) need = rows_n;
    rs_ready = !mv_pend || int'(y_ready) >= need;
    y_re = (st == V_RES || st == V_ARG) && i < cn && rs_ready;
    y_addr = (st == V_RES && cfl[2]) ? 2 * i : i;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= V_IDLE; done <= 1'b0; mv_start <= 1'b0; mv_pend <= 1'b0; x_we <= 1'b0;
      e1_v <= 1'b0; q1_v <= 1'b0; r1_v <= 1'b0; a1_v <= 1'b0; i <= '0; rms_go <= 1'b0;
      best <= '0; besti <= '0; a_first <= 1'b0; xsh <= '0;
    end else begin
      done <= 1'b0;
      mv_start <= 1'b0;
      rms_go <= 1'b0;
      e1_v <= 1'b0;
      q1_v <= 1'b0;
      r1_v <= 1'b0;
      a1_v <= 1'b0;
      if (mv_done) mv_pend <= 1'b0;
      case (st)
        V_IDLE: if (go) begin
          cop <= op; ca <= fa; cb <= fb; cc <= fc; cn <= fn; cm <= fm; ck <= fk; cimm <= imm; cfl <= flags; ctok <= tok; i <= '0;
          case (op)
            OP_EMBED:   st <= V_EMB;
            OP_QMV:     st <= V_QSET;
            OP_RESCALE: st <= V_RSET;
            OP_ARGMAX:  begin st <= V_ARG; a_first <= 1'b1; end
            default:    st <= V_DRAIN;
          endcase
        end
        V_EMB: if (i < cn) begin
          e1_v <= 1'b1; e1_i <= i; i <= i + VW;
        end else st <= V_DRAIN;
        V_QSET: if (!mv_pend && !wr_busy) begin
          int r;
          r = (shmax - DMAX > shmin) ? shmax - DMAX : shmin;
          ref_sh <= r;
          xsh <= 6'(r);
          if (cfl[0]) rms_go <= 1'b1;
          st <= V_QMV;
        end
        V_QMV: if (i < cn) begin
          q1_v <= 1'b1; q1_i <= i; i <= i + VW;
        end else st <= V_DRAIN;
        V_RSET: if (!cfl[0] || rms_ok) begin
          s_const <= int'(cimm) - int'(xsh) + (cfl[0] ? 7 + int'(rms_e) : 0);
          st <= V_RES;
        end
        V_RES: if (i >= cn) st <= V_DRAIN;
          else if (rs_ready) begin r1_v <= 1'b1; r1_i <= i; i <= i + VW; end
        V_ARG: if (i >= cn) st <= V_DRAIN;
          else if (rs_ready) begin a1_v <= 1'b1; a1_i <= i; i <= i + VW; end
        V_DRAIN: if (!pipe_busy && !wr_busy && !e1_v && !q1_v && !r1_v && !a1_v) begin
          st <= V_IDLE;
          done <= 1'b1;
        end
        default: st <= V_IDLE;
      endcase

      if (q1_v) begin
        for (int k = 0; k < VK; k++) begin
          logic [XW-1:0] word;
          int s;
          s = (int'(gsh[XA'((q1_i >> 5) + k)]) > ref_sh) ? int'(gsh[XA'((q1_i >> 5) + k)]) : ref_sh;
          for (int u = 0; u < G; u++) word[u*8 +: 8] = q8(vrd[(k*G+u)*48 +: 48], s);
          word[G*8 +: 5] = 5'(s - ref_sh);
          x_data[k] <= word;
          x_mask[k] <= (q1_i + 32'(k * G) < cn);
        end
        x_we <= 1'b1;
        x_addr <= q1_i >> 5;
        if (q1_i == 0) begin
          mv_base <= cb; mv_rows <= cm; mv_groups <= cn >> 5; mv_start <= 1'b1; mv_pend <= 1'b1;
        end
      end else x_we <= 1'b0;

      if (a_ov) begin
        if (a_first || $signed(a_max) > best) begin best <= $signed(a_max); besti <= a_imax; end
        a_first <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk) begin
    wi_v <= 1'b0;
    if (e1_v) begin
      for (int k = 0; k < VK; k++)
        for (int u = 0; u < G; u++) begin
          logic [WBITS-1:0] w;
          logic signed [15:0] code;
          logic signed [95:0] p;
          w = ewd[k][u*WBITS +: WBITS];
          code = (WBITS == 2) ? ((w == 2'b01) ? 16'sd1 : (w == 2'b11) ? -16'sd1 : 16'sd0) : 16'($signed(w));
          p = 96'(code * $signed({1'b0, ewd[k][G*WBITS +: 16]}));
          wi_d[k*G+u] <= rnd48(p, int'(cimm) - 16);
        end
      wi_v <= 1'b1;
      wi_row <= 32'(cc / VW) + e1_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (e1_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'b1; wi_first <= (e1_i == 0); wi_g0 <= e1_i >> 5;
    end
    if (r3_v && cfl[1]) begin
      for (int u = 0; u < VW; u++) wi_d[u] <= r3_x[u] + v48'(vrd[u*48 +: 48]);
      wi_v <= 1'b1;
      wi_row <= 32'(cc / VW) + r3_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (r3_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'b1; wi_first <= (r3_i == 0); wi_g0 <= r3_i >> 5;
    end
    if (r7_v) begin
      for (int u = 0; u < VW; u++) begin
        logic signed [99:0] t;
        t = 100'(r7_p0[u]) + (100'(r7_p1[u]) <<< 16) + (100'(r7_p2[u]) <<< 32) + 100'(1 << 15);
        wi_d[u] <= 48'(t >>> 16);
      end
      wi_v <= 1'b1;
      wi_row <= 32'(cc / VW) + r7_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (r7_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'b0; wi_first <= (r7_i == 0); wi_g0 <= r7_i >> 5;
    end
    if (o_we) begin
      wi_d <= o_data;
      wi_v <= 1'b1;
      wi_row <= o_row;
      wi_m <= o_mask;
      wi_trk <= 1'b1; wi_ss <= 1'b0; wi_first <= o_first; wi_g0 <= o_g0;
    end
  end

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {r2_v, r3_v, r4_v, r5_v, r6_v, r7_v, w1_v, w2_v, w3_v} <= '0;
    else begin
      r2_v <= r1_v;
      r3_v <= r2_v;
      r4_v <= r3_v && cfl[2];
      r5_v <= r4_v;
      r6_v <= r5_v;
      r7_v <= r6_v;
      w1_v <= wi_v;
      w2_v <= w1_v && w1_ss;
      w3_v <= w2_v;
    end

  always_ff @(posedge clk) begin
    q_we <= 1'b0;
    if (r1_v) begin
      r2_i <= r1_i;
      for (int u = 0; u < 2 * VW; u++)
        r2_y[u] <= cfl[0] ? 80'(y_vec[u] * $signed({1'b0, rms_l})) : 80'(y_vec[u]);
    end
    if (r2_v) begin
      r3_i <= r2_i;
      for (int u = 0; u < 2 * VW; u++) r3_x[u] <= rnd48(96'(r2_y[u]), s_const);
    end
    if (r3_v && !cfl[1] && !cfl[2]) begin
      q_we <= 1'b1;
      q_row <= r3_i / VW;
      for (int u = 0; u < VW; u++) q_data[u] <= r3_x[u];
      for (int k = 0; k < VK; k++) q_mask[k] <= (r3_i + 32'(k * G) < cn);
    end
  end

  always_ff @(posedge clk) begin
    if (r3_v && cfl[2]) begin
      r4_i <= r3_i;
      for (int u = 0; u < VW; u++) begin
        v48 g;
        g = r3_x[2*u];
        r4_g[u] <= g;
        r4_u[u] <= r3_x[2*u+1];
        r4_hi[u] <= g >= 48'sd524288;
        r4_lo[u] <= g < -48'sd524288;
        sigq[u] <= t_sig[10'((g + 48'sd524288) >>> 10)];
      end
    end
    if (r4_v) begin
      r5_i <= r4_i;
      for (int u = 0; u < VW; u++) begin
        logic [16:0] sg;
        sg = r4_hi[u] ? 17'd65536 : r4_lo[u] ? 17'd0 : sigq[u];
        r5_gs[u] <= 66'(r4_g[u] * $signed({1'b0, sg}));
        r5_u[u] <= r4_u[u];
      end
    end
    if (r5_v) begin
      r6_i <= r5_i;
      for (int u = 0; u < VW; u++) begin
        r6_s[u] <= 50'((r5_gs[u] + 66'(1 << 15)) >>> 16);
        r6_u[u] <= r5_u[u];
      end
    end
    if (r6_v) begin
      r7_i <= r6_i;
      for (int u = 0; u < VW; u++) begin
        r7_p0[u] <= 66'(r6_s[u] * $signed({1'b0, r6_u[u][15:0]}));
        r7_p1[u] <= 66'(r6_s[u] * $signed({1'b0, r6_u[u][31:16]}));
        r7_p2[u] <= 66'(r6_s[u] * $signed(r6_u[u][47:32]));
      end
    end
  end

  always_comb
    for (int u = 0; u < VW; u++) begin
      a_d[u] = (a1_i + 32'(u) < cn) ? y_vec[u] : {1'b1, 63'd0};
      a_ix[u] = a1_i + 32'(u);
    end
  reduce #(.N(VW), .W(64), .OP(2), .IW(32)) u_amax (
    .clk, .iv(a1_v), .din(a_d), .iin(a_ix), .ov(a_ov), .dout(a_max), .iout(a_imax));
  localparam int AL = ($clog2(VW) + 1) / 2;
  logic [AL:0] a_dl;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) a_dl <= '0;
    else a_dl <= {a_dl[AL-1:0], a1_v};
  assign a_busy = |a_dl;

  logic [143:0] rp, racc;
  logic [127:0] rmean;
  int rstep;
  int re;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin rms_busy <= 1'b0; rstep <= 0; rms_ok <= 1'b0; end
    else begin
      if (rms_go) begin rms_busy <= 1'b1; rstep <= 0; racc <= '0; rms_ok <= 1'b0; end
      else if (rms_busy) begin
        rstep <= rstep + 1;
        if (rstep < 4) rp <= 144'(ss * ck[rstep*8 +: 8]);
        if (rstep >= 1 && rstep <= 4) racc <= racc + (rp << (8 * (rstep - 1)));
        if (rstep == 5) rmean <= 128'(racc >> 32) + 128'(EPS);
        if (rstep == 6) re <= (bitlen(rmean) - 1 - 16) >>> 1;
        if (rstep == 7) begin
          logic [127:0] mm;
          mm = (re >= 0) ? (rmean >> (2 * re)) : (rmean << (-2 * re));
          rms_l <= t_rsqrt[int'(mm >> 10) - 64];
          rms_e <= 8'(re);
          rms_ok <= 1'b1;
          rms_busy <= 1'b0;
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (wi_v) begin
      for (int k = 0; k < VK; k++)
        if (wi_m[k]) vmem[VA'(wi_row)][k*G*48 +: G*48] <= {wi_d[k*G+31], wi_d[k*G+30], wi_d[k*G+29], wi_d[k*G+28], wi_d[k*G+27],
          wi_d[k*G+26], wi_d[k*G+25], wi_d[k*G+24], wi_d[k*G+23], wi_d[k*G+22], wi_d[k*G+21], wi_d[k*G+20], wi_d[k*G+19],
          wi_d[k*G+18], wi_d[k*G+17], wi_d[k*G+16], wi_d[k*G+15], wi_d[k*G+14], wi_d[k*G+13], wi_d[k*G+12], wi_d[k*G+11],
          wi_d[k*G+10], wi_d[k*G+9], wi_d[k*G+8], wi_d[k*G+7], wi_d[k*G+6], wi_d[k*G+5], wi_d[k*G+4], wi_d[k*G+3],
          wi_d[k*G+2], wi_d[k*G+1], wi_d[k*G]};
      for (int u = 0; u < VW; u++) w1_a[u] <= abs48(wi_d[u]);
      w1_m <= wi_m; w1_trk <= wi_trk; w1_ss <= wi_ss; w1_first <= wi_first; w1_g0 <= wi_g0;
    end
    if (w1_v && w1_ss) begin
      for (int u = 0; u < VW; u++) begin
        logic [15:0] a0, a1, a2;
        {a2, a1, a0} = w1_a[u];
        w2_p[u][0] <= a0 * a0; w2_p[u][1] <= a1 * a1; w2_p[u][2] <= a2 * a2;
        w2_p[u][3] <= a0 * a1; w2_p[u][4] <= a0 * a2; w2_p[u][5] <= a1 * a2;
      end
      w2_m <= w1_m; w2_first <= w1_first;
    end
    if (w2_v) begin
      for (int u = 0; u < VW; u++)
        w3_sq[u] <= w2_m[u / G] ? SQW'(w2_p[u][0]) + (SQW'(w2_p[u][3]) << 17) + (SQW'(w2_p[u][1]) << 32)
                                  + (SQW'(w2_p[u][4]) << 33) + (SQW'(w2_p[u][5]) << 49) + (SQW'(w2_p[u][2]) << 64) : '0;
      w3_first <= w2_first;
    end
  end

  always_comb for (int k = 0; k < VK; k++) for (int u = 0; u < G; u++) gin[k][u] = w1_a[k*G+u];
  mreduce #(.M(VK), .N(G), .W(48), .OP(1)) u_gmax (.clk, .iv(w1_v), .din(gin), .ov(gm_v), .dout(gm));
  always_comb for (int u = 0; u < VW; u++) zi[u] = '0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) for (int k = 0; k <= GL; k++) md_v[k] <= 1'b0;
    else begin
      md_v[0] <= w1_v;
      for (int k = 1; k <= GL; k++) md_v[k] <= md_v[k-1];
    end
  end
  always_ff @(posedge clk) begin
    md_m[0] <= w1_m; md_trk[0] <= w1_trk; md_first[0] <= w1_first; md_g0[0] <= w1_g0;
    for (int k = 1; k <= GL; k++) begin md_m[k] <= md_m[k-1]; md_trk[k] <= md_trk[k-1]; md_first[k] <= md_first[k-1]; md_g0[k] <= md_g0[k-1]; end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) w5_v <= 1'b0;
    else begin
      w5_v <= gm_v;
      if (gm_v) begin
        for (int k = 0; k < VK; k++) begin w5_gm[k] <= gm[k]; w5_s0[k] <= sh_a(gm[k]); end
        w5_m <= md_m[GL-1]; w5_trk <= md_trk[GL-1]; w5_first <= md_first[GL-1]; w5_g0 <= md_g0[GL-1];
      end
    end
  end

  always_ff @(posedge clk) begin
    if (w5_v && w5_trk) begin
      int mn, mx;
      mn = w5_first ? 63 : shmin;
      mx = w5_first ? 0 : shmax;
      for (int k = 0; k < VK; k++)
        if (w5_m[k]) begin
          int sh;
          sh = sh_b(w5_gm[k], w5_s0[k]);
          gsh[XA'(w5_g0 + 32'(k))] <= 6'(sh);
          if (sh < mn) mn = sh;
          if (sh > mx) mx = sh;
        end
      shmin <= mn;
      shmax <= mx;
    end
  end

  reduce #(.N(VW), .W(SQW), .OP(0), .LPS(2)) u_ss (
    .clk, .iv(w3_v), .din(w3_sq), .iin(zi), .ov(ss_ov), .dout(ss_chunk), .iout());
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) for (int k = 0; k <= SL; k++) sd_v[k] <= 1'b0;
    else begin
      sd_v[0] <= w3_v;
      for (int k = 1; k <= SL; k++) sd_v[k] <= sd_v[k-1];
    end
  end
  always_ff @(posedge clk) begin
    sd_first[0] <= w3_first;
    for (int k = 1; k <= SL; k++) sd_first[k] <= sd_first[k-1];
    if (ss_ov) ss <= (sd_first[SL-1] ? '0 : ss) + SSW'(ss_chunk);
  end

endmodule
