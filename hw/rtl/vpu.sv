module vpu #(
  parameter int VK     = 4,
  parameter int WBITS  = 4,
  parameter int VDEPTH = 4096,
  parameter int EDEPTH = 1024,
  parameter int XDEPTH = 32,
  parameter int ACT    = 0,
  parameter int SUBNORM = 0,
  parameter int GAINS  = 0,
  parameter int GDEPTH = 1,
  parameter int EPS    = 43
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   emb_we,
  input  logic [31:0]            emb_addr,
  input  logic [32*WBITS+15:0]   emb_data,
  input  logic                   tbl_we,
  input  logic [3:0]             tbl_sel,
  input  logic [31:0]            tbl_addr,
  input  logic [31:0]            tbl_data,
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
  output logic                   mv_wide,
  output logic [31:0]            mv_sbase,
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
  input  logic [31:0]            o_g0,
  input  logic [31:0]            dnu_ra,
  output logic [32*VK*48-1:0]    dnu_rd
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
  localparam int SQW   = 96 + $clog2(VW) + 1;
  localparam int SSW   = 112;
  localparam int GL    = 3;
  localparam int SL    = ($clog2(VW) + 1) / 2;
  localparam int GA    = GDEPTH > 1 ? $clog2(GDEPTH) : 1;
  localparam logic [7:0] OP_EMBED = 1, OP_QMV = 2, OP_RESCALE = 3, OP_ARGMAX = 5, OP_GAIN = 6, OP_HNORM = 7, OP_GATE = 8, OP_FEED = 9;

  typedef logic signed [47:0] v48;

  function automatic logic [47:0] abs48(input v48 v);
    abs48 = v[47] ? 48'(-v) : 48'(v);
  endfunction

  function automatic int bitlen(input logic [127:0] v);
    bitlen = 0;
    for (int k = 0; k < 128; k++) if (v[k]) bitlen = k + 1;
  endfunction

  function automatic logic [5:0] sh_a(input logic [47:0] m);
    logic [5:0] b;
    b = 6'(bitlen({80'd0, m}));
    sh_a = (b > 6'd7) ? b - 6'd7 : 6'd0;
  endfunction

  function automatic logic [5:0] sh_b(input logic [47:0] m, input logic [5:0] s0);
    logic [47:0] w;
    w = (s0 != 0) ? m >> (s0 - 6'd1) : '0;
    sh_b = (s0 != 0 && w[7:0] == 8'hff) ? s0 + 6'd1 : s0;
  endfunction

  function automatic int bitlen8(input logic [7:0] v);
    bitlen8 = 0;
    for (int k = 0; k < 8; k++) if (v[k]) bitlen8 = k + 1;
  endfunction

  function automatic logic signed [7:0] clamp8(input int v);
    clamp8 = (v > 100) ? 8'sd100 : (v < -100) ? -8'sd100 : 8'(v);
  endfunction

  function automatic v48 rnde(input logic signed [24:0] v, input logic signed [7:0] n);
    logic signed [25:0] t;
    if (n <= 0) rnde = 48'(v) <<< (-n);
    else if (n >= 25) rnde = '0;
    else begin
      t = 26'(v) + (26'(1) <<< (n - 1));
      rnde = 48'(t >>> n);
    end
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
    if (tbl_we && tbl_sel == 4'd0) t_rsqrt[tbl_addr] <= tbl_data[15:0];
    if (tbl_we && tbl_sel == 4'd2) t_sig[tbl_addr] <= tbl_data[16:0];
  end
  assign vm_rdata = vmem[VA'(vm_raddr / VW)][(vm_raddr % VW) * 48 +: 48];
  always_ff @(posedge clk) dnu_rd <= vmem[VA'(dnu_ra)];

  logic [31:0] hh;
  logic [1:0]  hsub, gp;
  logic        h1_v, h3_v, hn_load, fd_v, gt_v;
  logic [31:0] h3_i, fd_i, gt_i;
  logic [127:0] hss;
  logic        hq_v, hq_ov, h4_v;
  logic [103:0] hq [VW], hq_out;
  logic [31:0] hpend, h4_i;
  v48          hy [VW];
  logic signed [15:0] hg [VW];
  v48          gx [VW];
  logic [16:0] gsg [VW];
  logic        ghi [VW], glo [VW];

  typedef enum logic [3:0] {V_IDLE, V_EMB, V_QSET, V_QMV, V_RSET, V_RES, V_ARG, V_DRAIN, V_GAIN, V_HN1, V_HN2, V_HN3, V_GT, V_FD} vstate_t;
  vstate_t st;
  logic [7:0]  cop, cfl;
  logic [31:0] ca, cb, cc, cn, cm, ck, ctok;
  logic signed [15:0] cimm;
  logic [31:0] i;
  logic        mv_pend, rms_ok;
  logic signed [7:0] s_const, e_n;
  logic [23:0] e_rc;
  logic signed [96:0] rcon;
  logic [5:0]  shmin, shmax;
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

  logic        q1_v, q2_v;
  logic [31:0] q1_i, q2_i;
  logic [5:0]  qs [VK];
  logic signed [49:0] q2_h [VW];
  logic [4:0]  q2_e [VK];
  logic        e2_v;
  logic [31:0] e2_i;
  logic signed [25:0] e2_p [VW];
  logic        g1_v, g2_v;
  logic [31:0] g1_i, g2_i;
  logic signed [15:0] grd [VW];
  logic signed [63:0] g_p [VW];
  logic [VW*48-1:0] vrd;
  logic [31:0] vra;

  logic        r1_v, r2_v, r3_v, r4_v, r5_v, r6_v, r7_v;
  logic [31:0] r1_i, r2_i, r3_i, r4_i, r5_i, r6_i, r7_i;
  logic [48:0] r2_lo [2*VW];
  logic signed [48:0] r2_hi [2*VW];
  logic signed [96:0] r2_t [2*VW];
  logic        r2b_v;
  logic [31:0] r2b_i;
  v48          r3_x [2*VW];
  v48          r4_g [VW], r4_u [VW], r5_u [VW], r6_u [VW];
  logic        r4_hi [VW], r4_lo [VW];
  logic [16:0] sigq [VW];
  localparam int GW = ACT ? 97 : 66;
  localparam int SW = ACT ? 82 : 50;
  localparam int PW = SW + 17;
  localparam int TW = PW + 34;
  logic signed [GW-1:0] r5_gs [VW];
  logic signed [64:0]   r5_p0 [VW], r5_p1 [VW], r5_p2 [VW];
  logic signed [SW-1:0] r6_s [VW];
  logic signed [PW-1:0] r7_p0 [VW], r7_p1 [VW], r7_p2 [VW];

  logic        a1_v, a2_v;
  logic [63:0] a2_d [VW];
  logic [31:0] a2_ix [VW];
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

  logic        w1_v, w2_v, w2b_v, w3_v;
  logic [VK-1:0] w2b_m;
  logic        w2b_first;
  logic [SQW-1:0] w2b_a [VW], w2b_b [VW];
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
  logic [5:0]  w5_s0 [VK];
  logic        w6_v, w6_trk, w6_first;
  logic [VK-1:0] w6_m;
  logic [5:0]  w6_sh [VK];
  logic [SQW-1:0] ss_chunk;
  logic        ss_ov;
  logic [0:0]  zi [VW];

  logic wr_busy;
  always_comb begin
    wr_busy = wi_v | w1_v | w2_v | w2b_v | w3_v | w5_v | w6_v;
    for (int k = 0; k <= GL; k++) wr_busy |= md_v[k];
    for (int k = 0; k <= SL; k++) wr_busy |= sd_v[k];
  end
  logic a_busy, pipe_busy;
  assign pipe_busy = e1_v | e2_v | q1_v | q2_v | g1_v | g2_v | r1_v | r2_v | r2b_v | a2_v | r3_v | r4_v | r5_v | r6_v | r7_v | a1_v | a_busy | rms_busy
                     | h1_v | h3_v | fd_v | gt_v | hq_v | h4_v | (hpend != 0);
  assign idle = (st == V_IDLE) && !wr_busy && !pipe_busy;
  assign tok_out = besti;

  always_comb begin
    vra = 32'(cc / VW) + r2b_i / VW;
    if (st == V_QMV || st == V_GAIN || st == V_FD) vra = 32'(ca / VW) + i / VW;
    if (st == V_HN1 || st == V_HN3) vra = (ca + hh * cm) / VW + i / VW;
    if (st == V_GT) vra = ((gp == 2'd1) ? cb : ca) / VW + i / VW;
  end
  always_ff @(posedge clk)
    if (st == V_QMV || st == V_GAIN || st == V_HN1 || st == V_HN3 || st == V_GT || st == V_FD || r2b_v) vrd <= vmem[VA'(vra)];

  for (genvar u = 0; u < VW; u++) begin : g_gain
    logic [15:0] gm [GDEPTH];
    always_ff @(posedge clk) begin
      if (tbl_we && tbl_sel == 4'd5 && 32'(tbl_addr % VW) == 32'(u)) gm[GA'(tbl_addr / VW)] <= tbl_data[15:0];
      if (st == V_GAIN || st == V_HN3) grd[u] <= gm[GA'(cb / VW + i / VW)];
    end
  end

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
      e1_v <= 1'b0; q1_v <= 1'b0; q2_v <= 1'b0; g1_v <= 1'b0; r1_v <= 1'b0; a1_v <= 1'b0; i <= '0; rms_go <= 1'b0; mv_wide <= 1'b0;
      h1_v <= 1'b0; h3_v <= 1'b0; hn_load <= 1'b0; fd_v <= 1'b0; gt_v <= 1'b0; hh <= '0; hsub <= '0; gp <= '0; hpend <= '0;
      best <= '0; besti <= '0; a_first <= 1'b0; xsh <= '0;
    end else begin
      done <= 1'b0;
      mv_start <= 1'b0;
      rms_go <= 1'b0;
      e1_v <= 1'b0;
      q1_v <= 1'b0;
      g1_v <= 1'b0;
      r1_v <= 1'b0;
      a1_v <= 1'b0;
      h1_v <= 1'b0;
      h3_v <= 1'b0;
      hn_load <= 1'b0;
      fd_v <= 1'b0;
      gt_v <= 1'b0;
      hpend <= hpend + ((st == V_HN1 && i < cm) ? 32'd1 : 32'd0) - (hq_ov ? 32'd1 : 32'd0);
      if (mv_done) mv_pend <= 1'b0;
      case (st)
        V_IDLE: if (go) begin
          cop <= op; ca <= fa; cb <= fb; cc <= fc; cn <= fn; cm <= fm; ck <= fk; cimm <= imm; cfl <= flags; ctok <= tok; i <= '0;
          case (op)
            OP_EMBED:   begin
              st <= V_EMB;
              e_n <= clamp8(int'(imm) - 16);
              e_rc <= (int'(imm) - 16 > 0 && int'(imm) - 16 < 25) ? 24'(1) << (int'(imm) - 17) : '0;
            end
            OP_QMV:     st <= V_QSET;
            OP_RESCALE: st <= V_RSET;
            OP_ARGMAX:  begin st <= V_ARG; a_first <= 1'b1; end
            OP_GAIN:    st <= V_GAIN;
            OP_HNORM:   begin st <= V_HN1; hh <= '0; end
            OP_GATE:    begin st <= V_GT; gp <= '0; end
            OP_FEED:    st <= V_FD;
            default:    st <= V_DRAIN;
          endcase
        end
        V_EMB: if (i < cn) begin
          e1_v <= 1'b1; e1_i <= i; i <= i + VW;
        end else st <= V_DRAIN;
        V_QSET: if (!mv_pend && !wr_busy) begin
          xsh <= ({1'b0, shmax} > {1'b0, shmin} + 7'(DMAX)) ? shmax - 6'(DMAX) : shmin;
          if (cfl[0]) rms_go <= 1'b1;
          st <= V_QMV;
        end
        V_QMV: if (i < cn) begin
          q1_v <= 1'b1; q1_i <= i; i <= i + VW;
          for (int k = 0; k < VK; k++) qs[k] <= (gsh[XA'((i >> 5) + k)] > xsh) ? gsh[XA'((i >> 5) + k)] : xsh;
        end else st <= V_DRAIN;
        V_GAIN: if (i < cn) begin
          g1_v <= 1'b1; g1_i <= i; i <= i + VW;
        end else st <= V_DRAIN;
        V_HN1: if (i < cm) begin
          h1_v <= 1'b1; i <= i + VW;
        end else if (!h1_v && hpend == 0) begin st <= V_HN2; hsub <= '0; end
        V_HN2: begin
          if (hsub == 2'd0) begin hn_load <= 1'b1; hsub <= 2'd1; end
          else if (hsub == 2'd1) begin rms_go <= 1'b1; hsub <= 2'd2; end
          else if (hsub == 2'd2) hsub <= 2'd3;
          else if (rms_ok && !rms_busy) begin st <= V_HN3; i <= '0; end
        end
        V_HN3: if (i < cm) begin
          h3_v <= 1'b1; h3_i <= i; i <= i + VW;
        end else if (!h3_v && !h4_v) begin
          i <= '0;
          if ((hh + 1) * cm >= cn) st <= V_DRAIN;
          else begin hh <= hh + 1; st <= V_HN1; end
        end
        V_GT: if (i >= cn) begin if (!gt_v) st <= V_DRAIN; end
          else begin
            gp <= gp + 2'd1;
            if (gp == 2'd3) begin gt_v <= 1'b1; gt_i <= i; i <= i + VW; end
          end
        V_FD: if (i < cn) begin
          fd_v <= 1'b1; fd_i <= i; i <= i + VW;
        end else st <= V_DRAIN;
        V_RSET: if (!cfl[0] || rms_ok) begin
          int sc;
          sc = int'(cimm) - int'(xsh) + (cfl[0] ? 7 + int'(rms_e) : 0);
          s_const <= clamp8(sc);
          rcon <= (sc > 0 && sc < 96) ? (97'sd1 <<< (sc - 1)) : '0;
          st <= V_RES;
        end
        V_RES: if (i >= cn) st <= V_DRAIN;
          else if (rs_ready) begin r1_v <= 1'b1; r1_i <= i; i <= i + VW; end
        V_ARG: if (i >= cn) st <= V_DRAIN;
          else if (rs_ready) begin a1_v <= 1'b1; a1_i <= i; i <= i + VW; end
        V_DRAIN: if (!pipe_busy && !wr_busy && !r1_v && !a1_v) begin
          st <= V_IDLE;
          done <= 1'b1;
        end
        default: st <= V_IDLE;
      endcase

      q2_v <= q1_v;
      if (q1_v) begin
        q2_i <= q1_i;
        for (int k = 0; k < VK; k++) begin
          q2_e[k] <= 5'(qs[k] - xsh);
          for (int u = 0; u < G; u++) begin
            logic signed [49:0] x;
            x = 50'($signed(vrd[(k*G+u)*48 +: 48]));
            q2_h[k*G+u] <= (qs[k] == 0) ? x <<< 1 : x >>> (qs[k] - 6'd1);
          end
        end
      end
      if (q2_v) begin
        for (int k = 0; k < VK; k++) begin
          logic [XW-1:0] word;
          for (int u = 0; u < G; u++) begin
            logic signed [49:0] t;
            t = (q2_h[k*G+u] + 50'sd1) >>> 1;
            word[u*8 +: 8] = (t > 127) ? 8'sd127 : (t < -127) ? -8'sd127 : t[7:0];
          end
          word[G*8 +: 5] = q2_e[k];
          x_data[k] <= word;
          x_mask[k] <= (q2_i + 32'(k * G) < cn);
        end
        x_we <= 1'b1;
        x_addr <= q2_i >> 5;
        if (q2_i == 0) begin
          mv_base <= cb; mv_rows <= cm; mv_groups <= cn >> 5; mv_start <= 1'b1; mv_pend <= 1'b1; mv_wide <= cfl[3]; mv_sbase <= cc;
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
      e2_i <= e1_i;
      for (int k = 0; k < VK; k++)
        for (int u = 0; u < G; u++) begin
          logic [WBITS-1:0] w;
          logic signed [8:0] code;
          w = ewd[k][u*WBITS +: WBITS];
          code = (WBITS == 2) ? ((w == 2'b01) ? 9'sd1 : (w == 2'b11) ? -9'sd1 : 9'sd0) : 9'($signed(w));
          e2_p[k*G+u] <= 26'(code * $signed({1'b0, ewd[k][G*WBITS +: 16]})) + 26'(e_rc);
        end
    end
    if (e2_v) begin
      for (int u = 0; u < VW; u++) wi_d[u] <= (e_n <= 0) ? 48'(e2_p[u]) <<< (-e_n) : (e_n >= 25) ? '0 : 48'(e2_p[u] >>> e_n);
      wi_v <= 1'b1;
      wi_row <= 32'(cc / VW) + e2_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (e2_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'b1; wi_first <= (e2_i == 0); wi_g0 <= e2_i >> 5;
    end
    if (r3_v && cfl[1]) begin
      for (int u = 0; u < VW; u++) wi_d[u] <= r3_x[u] + v48'(vrd[u*48 +: 48]);
      wi_v <= 1'b1;
      wi_row <= 32'(cc / VW) + r3_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (r3_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'b1; wi_first <= (r3_i == 0); wi_g0 <= r3_i >> 5;
    end
    if (r3_v && cfl[4]) begin
      for (int u = 0; u < VW; u++) wi_d[u] <= r3_x[u];
      wi_v <= 1'b1;
      wi_row <= 32'(cc / VW) + r3_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (r3_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'b0; wi_first <= (r3_i == 0); wi_g0 <= r3_i >> 5;
    end
    if (r7_v) begin
      for (int u = 0; u < VW; u++) begin
        logic signed [TW-1:0] t;
        t = TW'(r7_p0[u]) + (TW'(r7_p1[u]) <<< 16) + (TW'(r7_p2[u]) <<< 32) + TW'(1 << 15);
        wi_d[u] <= 48'(t >>> 16);
      end
      wi_v <= 1'b1;
      wi_row <= 32'(cc / VW) + r7_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (r7_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'(SUBNORM); wi_first <= (r7_i == 0); wi_g0 <= r7_i >> 5;
    end
    if (g2_v) begin
      for (int u = 0; u < VW; u++) wi_d[u] <= rnd48(96'(g_p[u]), 15);
      wi_v <= 1'b1;
      wi_row <= 32'(cc / VW) + g2_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (g2_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'b0; wi_first <= (g2_i == 0); wi_g0 <= g2_i >> 5;
    end
    if (h4_v) begin
      for (int u = 0; u < VW; u++) wi_d[u] <= cfl[0] ? rnd48(96'(hy[u]) * 96'(hg[u]), int'(cimm)) : hy[u];
      wi_v <= 1'b1;
      wi_row <= (cc + hh * cm) / VW + h4_i / VW;
      wi_m <= '1;
      wi_trk <= 1'b1; wi_ss <= 1'b0; wi_first <= (hh == 0 && h4_i == 0); wi_g0 <= (hh * cm + h4_i) >> 5;
    end
    if (gt_v) begin
      for (int u = 0; u < VW; u++) begin
        logic [16:0] sg;
        sg = ghi[u] ? 17'd65536 : glo[u] ? 17'd0 : gsg[u];
        wi_d[u] <= rnd48(96'(gx[u]) * 96'($signed({1'b0, sg})), 16);
      end
      wi_v <= 1'b1;
      wi_row <= cc / VW + gt_i / VW;
      for (int k = 0; k < VK; k++) wi_m[k] <= (gt_i + 32'(k * G) < cn);
      wi_trk <= 1'b1; wi_ss <= 1'b0; wi_first <= (gt_i == 0); wi_g0 <= gt_i >> 5;
    end
    if (o_we) begin
      wi_d <= o_data;
      wi_v <= 1'b1;
      wi_row <= o_row;
      wi_m <= o_mask;
      wi_trk <= 1'b1; wi_ss <= 1'(SUBNORM); wi_first <= o_first; wi_g0 <= o_g0;
    end
  end

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {r2_v, r2b_v, r3_v, r4_v, r5_v, r6_v, r7_v, w1_v, w2_v, w2b_v, w3_v, g2_v, e2_v, a2_v} <= '0;
    else begin
      e2_v <= e1_v;
      a2_v <= a1_v;
      r2b_v <= r2_v;
      g2_v <= g1_v;
      r2_v <= r1_v;
      r3_v <= r2b_v;
      r4_v <= r3_v && cfl[2];
      r5_v <= r4_v;
      r6_v <= r5_v;
      r7_v <= r6_v;
      w1_v <= wi_v;
      w2_v <= w1_v && w1_ss;
      w2b_v <= w2_v;
      w3_v <= w2b_v;
    end

  always_ff @(posedge clk) begin
    if (st == V_HN1 && i == 0) hss <= '0;
    else if (hq_ov) hss <= hss + 128'(hq_out);
    if (h1_v) for (int u = 0; u < VW; u++) hq[u] <= 104'($signed(vrd[u*48 +: 48]) * $signed(vrd[u*48 +: 48]));
    h4_i <= h3_i;
    if (h3_v)
      for (int u = 0; u < VW; u++) begin
        hy[u] <= rnd48(96'($signed(vrd[u*48 +: 48])) * 96'($signed({1'b0, rms_l})), 7 + int'(rms_e));
        hg[u] <= grd[u];
      end
    if (st == V_GT && gp == 2'd1) for (int u = 0; u < VW; u++) gx[u] <= $signed(vrd[u*48 +: 48]);
    if (st == V_GT && gp == 2'd2)
      for (int u = 0; u < VW; u++) begin
        v48 g;
        g = $signed(vrd[u*48 +: 48]);
        ghi[u] <= g >= 48'sd524288;
        glo[u] <= g < -48'sd524288;
        gsg[u] <= t_sig[10'((g + 48'sd524288) >>> 10)];
      end
  end

  always_ff @(posedge clk)
    if (g1_v) begin
      g2_i <= g1_i;
      for (int u = 0; u < VW; u++)
        g_p[u] <= cfl[0] ? 64'($signed(vrd[u*48 +: 48])) <<< 15 : 64'($signed(vrd[u*48 +: 48]) * grd[u]);
    end

  always_ff @(posedge clk) begin
    q_we <= 1'b0;
    if (r1_v) begin
      r2_i <= r1_i;
      for (int u = 0; u < 2 * VW; u++) begin
        logic [16:0] m;
        m = cfl[0] ? {1'b0, rms_l} : 17'd1;
        r2_lo[u] <= 49'(y_vec[u][31:0] * m);
        r2_hi[u] <= 49'($signed(y_vec[u][63:32]) * $signed({1'b0, m}));
      end
    end
    if (r2_v) begin
      r2b_i <= r2_i;
      for (int u = 0; u < 2 * VW; u++) r2_t[u] <= (97'(r2_hi[u]) <<< 32) + 97'({1'b0, r2_lo[u]}) + rcon;
    end
    if (r2b_v) begin
      r3_i <= r2b_i;
      for (int u = 0; u < 2 * VW; u++)
        r3_x[u] <= (s_const <= 0) ? 48'(r2_t[u] <<< (-s_const)) : (s_const >= 96) ? '0 : 48'(r2_t[u] >>> s_const);
    end
    if (fd_v) begin
      q_we <= 1'b1;
      q_row <= fd_i / VW;
      for (int u = 0; u < VW; u++) q_data[u] <= $signed(vrd[u*48 +: 48]);
      for (int k = 0; k < VK; k++) q_mask[k] <= (fd_i + 32'(k * G) < cn);
    end
    if (r3_v && !cfl[1] && !cfl[2] && !cfl[4]) begin
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
        r4_g[u] <= (ACT != 0 && g < 0) ? '0 : g;
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
        if (ACT) begin
          r5_p0[u] <= 65'(r4_g[u] * $signed({1'b0, r4_g[u][15:0]}));
          r5_p1[u] <= 65'(r4_g[u] * $signed({1'b0, r4_g[u][31:16]}));
          r5_p2[u] <= 65'(r4_g[u] * $signed(r4_g[u][47:32]));
        end else r5_gs[u] <= GW'(r4_g[u] * $signed({1'b0, sg}));
        r5_u[u] <= r4_u[u];
      end
    end
    if (r5_v) begin
      r6_i <= r5_i;
      for (int u = 0; u < VW; u++) begin
        logic signed [GW-1:0] gs;
        gs = ACT ? GW'(r5_p0[u]) + (GW'(r5_p1[u]) <<< 16) + (GW'(r5_p2[u]) <<< 32) : r5_gs[u];
        r6_s[u] <= SW'((gs + GW'(1 << 15)) >>> 16);
        r6_u[u] <= r5_u[u];
      end
    end
    if (r6_v) begin
      r7_i <= r6_i;
      for (int u = 0; u < VW; u++) begin
        r7_p0[u] <= PW'(r6_s[u] * $signed({1'b0, r6_u[u][15:0]}));
        r7_p1[u] <= PW'(r6_s[u] * $signed({1'b0, r6_u[u][31:16]}));
        r7_p2[u] <= PW'(r6_s[u] * $signed(r6_u[u][47:32]));
      end
    end
  end

  always_comb
    for (int u = 0; u < VW; u++) begin
      a_d[u] = (a1_i + 32'(u) < cn) ? y_vec[u] : {1'b1, 63'd0};
      a_ix[u] = a1_i + 32'(u);
    end
  always_ff @(posedge clk) if (a1_v) begin a2_d <= a_d; a2_ix <= a_ix; end
  reduce #(.N(VW), .W(64), .OP(2), .IW(32)) u_amax (
    .clk, .iv(a2_v), .din(a2_d), .iin(a2_ix), .ov(a_ov), .dout(a_max), .iout(a_imax));
  localparam int AL = ($clog2(VW) + 1) / 2;
  logic [AL:0] a_dl;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) a_dl <= '0;
    else a_dl <= {a_dl[AL-1:0], a2_v};
  assign a_busy = |a_dl;

  logic [143:0] rp, racc;
  logic [127:0] rmean, rmm;
  int rstep;
  logic signed [7:0] re;
  logic [3:0]  rhi;
  logic [7:0]  rhb;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin rms_busy <= 1'b0; rstep <= 0; rms_ok <= 1'b0; end
    else begin
      if (rms_go) begin rms_busy <= 1'b1; rstep <= 0; racc <= '0; rms_ok <= 1'b0; end
      else if (rms_busy) begin
        rstep <= rstep + 1;
        if (rstep < 4) rp <= 144'(ss * ck[rstep*8 +: 8]);
        if (rstep >= 1 && rstep <= 4) racc <= racc + (rp << (8 * (rstep - 1)));
        if (rstep == 5) rmean <= 128'(racc >> 32) + 128'(EPS);
        if (rstep == 6) begin
          logic [3:0] hi;
          hi = 4'd0;
          for (int k = 1; k < 16; k++) if (rmean[k*8 +: 8] != 0) hi = 4'(k);
          rhi <= hi;
          rhb <= rmean[hi*8 +: 8];
        end
        if (rstep == 7) re <= 8'(((rhb == 0 ? 0 : int'(rhi) * 8 + bitlen8(rhb)) - 17) >>> 1);
        if (rstep == 8) rmm <= (re >= 0) ? (rmean >> (2 * int'(re))) : (rmean << (-2 * int'(re)));
        if (rstep == 9) begin
          rms_l <= t_rsqrt[int'(rmm >> 10) - 64];
          rms_e <= re;
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
      for (int u = 0; u < VW; u++) begin
        w2b_a[u] <= SQW'(w2_p[u][0]) + (SQW'(w2_p[u][3]) << 17) + (SQW'(w2_p[u][1]) << 32);
        w2b_b[u] <= (SQW'(w2_p[u][4]) << 33) + (SQW'(w2_p[u][5]) << 49) + (SQW'(w2_p[u][2]) << 64);
      end
      w2b_m <= w2_m; w2b_first <= w2_first;
    end
    if (w2b_v) begin
      for (int u = 0; u < VW; u++) w3_sq[u] <= w2b_m[u / G] ? w2b_a[u] + w2b_b[u] : '0;
      w3_first <= w2b_first;
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

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) w6_v <= 1'b0;
    else w6_v <= w5_v && w5_trk;

  always_ff @(posedge clk) begin
    if (w5_v && w5_trk) begin
      for (int k = 0; k < VK; k++) begin
        w6_sh[k] <= sh_b(w5_gm[k], w5_s0[k]);
        if (w5_m[k]) gsh[XA'(w5_g0 + 32'(k))] <= sh_b(w5_gm[k], w5_s0[k]);
      end
      w6_m <= w5_m; w6_first <= w5_first;
    end
    if (w6_v) begin
      logic [5:0] mn, mx;
      mn = w6_first ? 6'd63 : shmin;
      mx = w6_first ? 6'd0 : shmax;
      for (int k = 0; k < VK; k++)
        if (w6_m[k]) begin
          if (w6_sh[k] < mn) mn = w6_sh[k];
          if (w6_sh[k] > mx) mx = w6_sh[k];
        end
      shmin <= mn;
      shmax <= mx;
    end
  end

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) {hq_v, h4_v} <= '0;
    else begin hq_v <= h1_v; h4_v <= h3_v; end
  reduce #(.N(VW), .W(104), .OP(0), .LPS(2)) u_hq (.clk, .iv(hq_v), .din(hq), .iin(zi), .ov(hq_ov), .dout(hq_out), .iout());

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
    if (hn_load) ss <= SSW'(hss);
    else if (ss_ov) ss <= (sd_first[SL-1] ? '0 : ss) + SSW'(ss_chunk);
  end

endmodule
