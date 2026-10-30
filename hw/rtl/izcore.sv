module izcore #(
  parameter int LANES   = 8,
  parameter int WBITS   = 4,
  parameter int VK      = 1,
  parameter int HD      = 64,
  parameter int NKV     = 2,
  parameter int NLAYERS = 4,
  parameter int MAXSEQ  = 256,
  parameter int ATT_TOK = 8,
  parameter int REP     = 2,
  parameter int VDEPTH  = 4096,
  parameter int IDEPTH  = 512,
  parameter int EDEPTH  = 1024,
  parameter int WDEPTH  = 16384,
  parameter int XDEPTH  = 32,
  parameter int YDEPTH  = 2048
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   imem_we,
  input  logic [31:0]            imem_addr,
  input  logic [255:0]           imem_data,
  input  logic                   emb_we,
  input  logic [31:0]            emb_addr,
  input  logic [32*WBITS+15:0]   emb_data,
  input  logic                   w_we,
  input  logic [$clog2(LANES)-1:0] w_lane,
  input  logic [31:0]            w_addr,
  input  logic [32*WBITS+15:0]   w_data,
  input  logic                   tbl_we,
  input  logic [2:0]             tbl_sel,
  input  logic [31:0]            tbl_addr,
  input  logic [16:0]            tbl_data,
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
  localparam logic [7:0] OP_END = 0, OP_EMBED = 1, OP_QMV = 2, OP_RESCALE = 3, OP_ATTN = 4, OP_ARGMAX = 5;

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

  logic         vpu_go, attn_go, tok_start, vpu_done, vpu_idle, attn_done, udone;
  logic [31:0]  pc, tok, pos, vpu_tok;
  logic [63:0]  cyc;

  logic                  x_we, mv_start, mv_busy, mv_done, q_we, o_we, o_first;
  logic [VK-1:0]         x_mask, q_mask, o_mask;
  logic [31:0]           x_addr, mv_base, mv_rows, mv_groups, y_ready, y_addr, q_row, o_row, o_g0;
  logic                  y_re;
  logic [32*8+4:0]       x_data [VK];
  logic signed [63:0]    y_vec [2*VW];
  logic signed [47:0]    q_data [VW];
  logic signed [47:0]    o_data [VW];

  mvu #(.LANES(LANES), .WBITS(WBITS), .WDEPTH(WDEPTH), .XDEPTH(XDEPTH), .YDEPTH(YDEPTH), .YPORTS(2*VW), .XPORTS(VK)) u_mvu (
    .clk, .rst_n, .w_we, .w_lane, .w_addr, .w_data, .x_we, .x_mask, .x_addr, .x_data,
    .cfg_base(mv_base), .cfg_rows(mv_rows), .cfg_groups(mv_groups), .start(mv_start),
    .busy(mv_busy), .done(mv_done), .y_ready, .y_addr, .y_re, .y_vec);

  vpu #(.VK(VK), .WBITS(WBITS), .VDEPTH(VDEPTH), .EDEPTH(EDEPTH), .XDEPTH(XDEPTH)) u_vpu (
    .clk, .rst_n, .emb_we, .emb_addr, .emb_data, .tbl_we, .tbl_sel, .tbl_addr, .tbl_data, .vm_raddr, .vm_rdata,
    .go(vpu_go), .op, .fa, .fb, .fc, .fn, .fm, .fk, .imm, .flags, .tok, .done(vpu_done), .idle(vpu_idle), .tok_out(vpu_tok),
    .x_we, .x_mask, .x_addr, .x_data, .mv_start, .mv_base, .mv_rows, .mv_groups, .mv_done, .y_ready, .y_addr, .y_re, .y_vec,
    .q_we, .q_row, .q_data, .q_mask, .o_we, .o_row, .o_data, .o_mask, .o_first, .o_g0);

  attn #(.VK(VK), .HD(HD), .NKV(NKV), .REP(REP), .NLAYERS(NLAYERS), .MAXSEQ(MAXSEQ), .ATT_TOK(ATT_TOK)) u_attn (
    .clk, .rst_n, .tbl_we, .tbl_sel, .tbl_addr, .tbl_data, .tok_start, .pos, .q_we, .q_row, .q_data, .q_mask,
    .go(attn_go), .fc, .fk, .imm, .done(attn_done), .o_we, .o_row, .o_data, .o_mask, .o_first, .o_g0);

  typedef enum logic [1:0] {S_IDLE, S_FETCH, S_ISSUE, S_WAIT} state_t;
  state_t state;
  assign busy = (state != S_IDLE);
  assign tok_out = tok;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; pc <= '0; ir <= '0; done <= 1'b0; error <= 1'b0; vpu_go <= 1'b0; attn_go <= 1'b0; tok_start <= 1'b0;
      udone <= 1'b0; tok <= '0; pos <= '0; cyc <= '0; cycles <= '0;
      for (int k = 0; k < NOPS; k++) perf[k] <= '0;
    end else begin
      done <= 1'b0;
      vpu_go <= 1'b0;
      attn_go <= 1'b0;
      tok_start <= 1'b0;
      if (state != S_IDLE) cyc <= cyc + 1;
      if (state == S_ISSUE || state == S_WAIT) perf[op[3:0]] <= perf[op[3:0]] + 1;
      if (vpu_done || attn_done) udone <= 1'b1;
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
          else if (op != OP_ATTN && fn == 0) state <= S_FETCH;
          else if (op == OP_ATTN) begin attn_go <= 1'b1; state <= S_WAIT; end
          else if (op == OP_EMBED || op == OP_QMV || op == OP_RESCALE || op == OP_ARGMAX) begin vpu_go <= 1'b1; state <= S_WAIT; end
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
