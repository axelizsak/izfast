module izcore #(
  parameter int LANES   = 8,
  parameter int WBITS   = 4,
  parameter int HBITS   = WBITS,
  parameter int VK      = 1,
  parameter int HD      = 64,
  parameter int RD      = HD,
  parameter int NKV     = 2,
  parameter int NLAYERS = 4,
  parameter int MAXSEQ  = 256,
  parameter int ATT_TOK = 8,
  parameter int REP     = 2,
  parameter int VDEPTH  = 4096,
  parameter int IDEPTH  = 512,
  parameter int EDEPTH  = 1024,
  parameter int WDEPTH  = 16384,
  parameter int HDEPTH  = 1,
  parameter int XDEPTH  = 32,
  parameter int YDEPTH  = 2048,
  parameter int ACT     = 0,
  parameter int SUBNORM = 0,
  parameter int GAINS   = 0,
  parameter int GDEPTH  = 1,
  parameter int SDEPTH  = 1,
  parameter int EPS     = 43,
  parameter int LIN     = 0,
  parameter int HK      = 2,
  parameter int HV      = 4,
  parameter int DK      = 32,
  parameter int DV      = 32,
  parameter int CK      = 4,
  parameter int NLIN    = 1,
  parameter int SH      = 1,
  parameter int LS      = 5793,
  parameter int NWW     = (WBITS == 2) ? 56 : 32 * WBITS + 16,
  parameter int RW      = (HBITS != WBITS && 32 * HBITS + 16 > NWW) ? 32 * HBITS + 16 : NWW
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   imem_we,
  input  logic [31:0]            imem_addr,
  input  logic [255:0]           imem_data,
  input  logic                   emb_we,
  input  logic [31:0]            emb_addr,
  input  logic [32*HBITS+15:0]   emb_data,
  input  logic                   w_we,
  input  logic [$clog2(LANES)-1:0] w_lane,
  input  logic [31:0]            w_addr,
  input  logic [RW-1:0]          w_data,
  input  logic                   s_we,
  input  logic [$clog2(LANES)-1:0] s_lane,
  input  logic [31:0]            s_addr,
  input  logic [15:0]            s_data,
  input  logic                   tbl_we,
  input  logic [3:0]             tbl_sel,
  input  logic [31:0]            tbl_addr,
  input  logic [31:0]            tbl_data,
  input  logic [31:0]            vm_raddr,
  output logic signed [47:0]     vm_rdata,
  input  logic [15:0]            perf_sel,
  output logic [63:0]            perf_val,
  input  logic [31:0]            tok_in,
  input  logic [31:0]            pos_in,
  input  logic                   start,
  output logic                   busy,
  output logic                   done,
  output logic                   error,
  output logic [31:0]            tok_out,
  output logic [63:0]            cycles
);

  localparam int VW   = 32 * VK;
  localparam int IA   = $clog2(IDEPTH);
  localparam int NOPS = 16;
  localparam logic [7:0] OP_END = 0, OP_EMBED = 1, OP_QMV = 2, OP_RESCALE = 3, OP_ATTN = 4, OP_ARGMAX = 5, OP_GAIN = 6,
                         OP_HNORM = 7, OP_GATE = 8, OP_FEED = 9, OP_DELTA = 10;

  logic [255:0] imem [IDEPTH];
  logic [63:0]  perf [NOPS];
  always_ff @(posedge clk) if (imem_we) imem[IA'(imem_addr)] <= imem_data;
  assign perf_val = perf[perf_sel[3:0]];

  logic [255:0] ir;
  logic [7:0]   op, flags;
  logic [31:0]  fa, fb, fc, fn, fm, fk;
  logic signed [15:0] imm;
  assign op = ir[7:0];
  assign fa = ir[39:8];
  assign fb = ir[71:40];
  assign fc = ir[103:72];
  assign fn = ir[135:104];
  assign fm = ir[167:136];
  assign imm = ir[183:168];
  assign flags = ir[191:184];
  assign fk = ir[223:192];

  logic         vpu_go, attn_go, dnu_go, tok_start, vpu_done, vpu_idle, attn_done, dnu_done, udone;
  logic [31:0]  dnu_ra;
  logic [32*VK*48-1:0] dnu_rd;
  logic         a_we, d_we, a_first, d_first;
  logic [31:0]  a_row, d_row, a_g0, d_g0;
  logic [VK-1:0] a_mask, d_mask;
  logic signed [47:0] a_data [VW], d_data [VW];
  logic [31:0]  pc, tok, pos, vpu_tok;
  logic [63:0]  cyc;

  logic                  x_we, mv_start, mv_wide, mv_busy, mv_done, q_we, o_we, o_first;
  logic [31:0]           mv_sbase;
  logic [VK-1:0]         x_mask, q_mask, o_mask;
  logic [31:0]           x_addr, mv_base, mv_rows, mv_groups, y_ready, y_addr, q_row, o_row, o_g0;
  logic                  y_re;
  logic [32*8+4:0]       x_data [VK];
  logic signed [63:0]    y_vec [2*VW];
  logic signed [47:0]    q_data [VW];
  logic signed [47:0]    o_data [VW];

  mvu #(.LANES(LANES), .WBITS(WBITS), .HBITS(HBITS), .SDEPTH(SDEPTH), .WDEPTH(WDEPTH), .HDEPTH(HDEPTH), .XDEPTH(XDEPTH), .YDEPTH(YDEPTH), .YPORTS(2*VW), .XPORTS(VK)) u_mvu (
    .clk, .rst_n, .w_we, .w_lane, .w_addr, .w_data, .s_we, .s_lane, .s_addr, .s_data, .x_we, .x_mask, .x_addr, .x_data,
    .cfg_base(mv_base), .cfg_rows(mv_rows), .cfg_groups(mv_groups), .cfg_wide(mv_wide), .cfg_sbase(mv_sbase), .start(mv_start),
    .busy(mv_busy), .done(mv_done), .y_ready, .y_addr, .y_re, .y_vec);

  vpu #(.VK(VK), .WBITS(HBITS), .VDEPTH(VDEPTH), .EDEPTH(EDEPTH), .XDEPTH(XDEPTH), .ACT(ACT), .SUBNORM(SUBNORM),
        .GAINS(GAINS), .GDEPTH(GDEPTH), .EPS(EPS)) u_vpu (
    .clk, .rst_n, .emb_we, .emb_addr, .emb_data, .tbl_we, .tbl_sel, .tbl_addr, .tbl_data, .vm_raddr, .vm_rdata,
    .go(vpu_go), .op, .fa, .fb, .fc, .fn, .fm, .fk, .imm, .flags, .tok, .done(vpu_done), .idle(vpu_idle), .tok_out(vpu_tok),
    .x_we, .x_mask, .x_addr, .x_data, .mv_start, .mv_wide, .mv_sbase, .mv_base, .mv_rows, .mv_groups, .mv_done, .y_ready, .y_addr, .y_re, .y_vec,
    .q_we, .q_row, .q_data, .q_mask, .o_we, .o_row, .o_data, .o_mask, .o_first, .o_g0, .dnu_ra, .dnu_rd);

  attn #(.VK(VK), .HD(HD), .RD(RD), .NKV(NKV), .REP(REP), .NLAYERS(NLAYERS), .MAXSEQ(MAXSEQ), .ATT_TOK(ATT_TOK)) u_attn (
    .clk, .rst_n, .tbl_we, .tbl_sel, .tbl_addr, .tbl_data, .tok_start, .pos, .q_we, .q_row, .q_data, .q_mask,
    .go(attn_go), .fc, .fk, .imm, .done(attn_done), .o_we(a_we), .o_row(a_row), .o_data(a_data), .o_mask(a_mask), .o_first(a_first), .o_g0(a_g0));

  if (LIN) begin : g_dnu
    dnu #(.VK(VK), .HK(HK), .HV(HV), .DK(DK), .DV(DV), .CK(CK), .NLIN(NLIN), .SH(SH), .LS(LS), .EPS(EPS)) u_dnu (
      .clk, .rst_n, .tbl_we, .tbl_sel, .tbl_addr, .tbl_data, .pos, .go(dnu_go), .fa, .fc, .fk, .imm, .done(dnu_done),
      .rd_row(dnu_ra), .rd_data(dnu_rd), .o_we(d_we), .o_row(d_row), .o_data(d_data), .o_mask(d_mask), .o_first(d_first), .o_g0(d_g0));
  end else begin : g_nodnu
    assign dnu_done = 1'b0; assign dnu_ra = '0; assign d_we = 1'b0; assign d_row = '0; assign d_mask = '0; assign d_first = 1'b0; assign d_g0 = '0;
    for (genvar u = 0; u < VW; u++) begin : g_z
      assign d_data[u] = '0;
    end
  end

  assign o_we = a_we | d_we;
  assign o_row = d_we ? d_row : a_row;
  assign o_mask = d_we ? d_mask : a_mask;
  assign o_first = d_we ? d_first : a_first;
  assign o_g0 = d_we ? d_g0 : a_g0;
  for (genvar u = 0; u < VW; u++) begin : g_omux
    assign o_data[u] = d_we ? d_data[u] : a_data[u];
  end

  typedef enum logic [1:0] {S_IDLE, S_FETCH, S_ISSUE, S_WAIT} state_t;
  state_t state;
  assign busy = (state != S_IDLE);
  assign tok_out = tok;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; pc <= '0; ir <= '0; done <= 1'b0; error <= 1'b0; vpu_go <= 1'b0; attn_go <= 1'b0; dnu_go <= 1'b0; tok_start <= 1'b0;
      udone <= 1'b0; tok <= '0; pos <= '0; cyc <= '0; cycles <= '0;
      for (int k = 0; k < NOPS; k++) perf[k] <= '0;
    end else begin
      done <= 1'b0;
      vpu_go <= 1'b0;
      attn_go <= 1'b0;
      dnu_go <= 1'b0;
      tok_start <= 1'b0;
      if (state != S_IDLE) cyc <= cyc + 1;
      if (state == S_ISSUE || state == S_WAIT) perf[op[3:0]] <= perf[op[3:0]] + 1;
      if (vpu_done || attn_done || dnu_done) udone <= 1'b1;
      case (state)
        S_IDLE: if (start) begin
          if (pos_in >= MAXSEQ) begin error <= 1'b1; done <= 1'b1; end
          else begin
            state <= S_FETCH; pc <= '0; tok <= tok_in; pos <= pos_in; cyc <= 64'd1; error <= 1'b0; tok_start <= 1'b1;
            for (int k = 0; k < NOPS; k++) perf[k] <= '0;
          end
        end
        S_FETCH: begin
          ir <= imem[IA'(pc)];
          pc <= pc + 1;
          state <= S_ISSUE;
        end
        S_ISSUE: begin
          udone <= 1'b0;
          if (op == OP_END) begin state <= S_IDLE; done <= 1'b1; cycles <= cyc; end
          else if (op != OP_ATTN && op != OP_DELTA && fn == 0) state <= S_FETCH;
          else if (op == OP_ATTN) begin attn_go <= 1'b1; state <= S_WAIT; end
          else if (op == OP_DELTA && LIN != 0) begin dnu_go <= 1'b1; state <= S_WAIT; end
          else if (op == OP_EMBED || op == OP_QMV || op == OP_RESCALE || op == OP_ARGMAX || op == OP_GAIN || op == OP_HNORM || op == OP_GATE
                   || op == OP_FEED) begin vpu_go <= 1'b1; state <= S_WAIT; end
          else begin state <= S_IDLE; done <= 1'b1; error <= 1'b1; cycles <= cyc; end
        end
        S_WAIT: if (udone && vpu_idle) begin
          if (op == OP_ARGMAX) tok <= vpu_tok;
          ir <= imem[IA'(pc)];
          pc <= pc + 1;
          state <= S_ISSUE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
