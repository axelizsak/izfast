module mvu #(
  parameter int LANES   = 8,
  parameter int GROUP   = 32,
  parameter int WBITS   = 4,
  parameter int SBITS   = 16,
  parameter int ABITS   = 8,
  parameter int DBITS   = 5,
  parameter int DMAX    = 24,
  parameter int WDEPTH  = 16384,
  parameter int XDEPTH  = 512,
  parameter int YDEPTH  = 2048,
  parameter int YPORTS  = 32,
  parameter int XPORTS  = 1,
  parameter int ACCW    = 64
) (
  input  logic                                  clk,
  input  logic                                  rst_n,
  input  logic                                  w_we,
  input  logic [$clog2(LANES)-1:0]              w_lane,
  input  logic [31:0]                           w_addr,
  input  logic [GROUP*WBITS+SBITS-1:0]          w_data,
  input  logic                                  x_we,
  input  logic [XPORTS-1:0]                     x_mask,
  input  logic [31:0]                           x_addr,
  input  logic [GROUP*ABITS+DBITS-1:0]          x_data [XPORTS],
  input  logic [31:0]                           cfg_base,
  input  logic [31:0]                           cfg_rows,
  input  logic [31:0]                           cfg_groups,
  input  logic                                  start,
  output logic                                  busy,
  output logic                                  done,
  output logic [31:0]                           y_ready,
  input  logic [31:0]                           y_addr,
  input  logic                                  y_re,
  output logic signed [ACCW-1:0]                y_vec [YPORTS]
);

  localparam bit TERNARY = (WBITS == 2);
  localparam int WW     = GROUP * WBITS + SBITS;
  localparam int XW     = GROUP * ABITS + DBITS;
  localparam int LEVELS = $clog2(GROUP);
  localparam int PW     = WBITS + ABITS;
  localparam int DW     = PW + LEVELS;
  localparam int LB     = $clog2(LANES);
  localparam int WA     = $clog2(WDEPTH);
  localparam int XA     = $clog2(XDEPTH);
  localparam int YA     = $clog2(YDEPTH);

  logic [WW-1:0]          wmem [LANES][WDEPTH];
  logic [XW-1:0]          xmem [XDEPTH];
  logic signed [ACCW-1:0] ymem [YDEPTH];

`ifndef SYNTHESIS
  initial begin
    string f;
    if ($value$plusargs("wmem=%s", f)) $readmemh(f, wmem);
  end
`endif

  always_ff @(posedge clk) begin
    if (w_we) wmem[w_lane][w_addr[WA-1:0]] <= w_data;
    if (x_we)
      for (int k = 0; k < XPORTS; k++)
        if (x_mask[k]) xmem[XA'(x_addr + 32'(k))] <= x_data[k];
  end

  always_ff @(posedge clk)
    if (y_re) for (int u = 0; u < YPORTS; u++) y_vec[u] <= ymem[YA'(y_addr + 32'(u))];

  typedef enum logic [1:0] {IDLE, RUN, DRAIN} state_t;
  state_t state;
  logic [31:0] g, blk, nblk, groups, rows, addr, y_rdy;
  assign y_ready = start ? '0 : y_rdy;

  localparam int L1     = 2;
  localparam int N1     = GROUP >> L1;
  localparam int W1     = PW + L1;
  localparam int NTAG   = 5;

  typedef struct packed {
    logic        valid;
    logic        first;
    logic        last;
    logic [31:0] blk;
  } tag_t;
  tag_t tag [NTAG];

  logic issue, pipe_busy;
  always_comb begin
    pipe_busy = 1'b0;
    for (int k = 0; k < NTAG; k++) pipe_busy |= tag[k].valid;
  end
  assign issue = (state == RUN);
  assign busy  = (state != IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE; g <= '0; blk <= '0; nblk <= '0; groups <= '0; rows <= '0; addr <= '0; done <= 1'b0; y_rdy <= '0;
    end else begin
      done <= 1'b0;
      if (tag[NTAG-1].valid && tag[NTAG-1].last)
        y_rdy <= (((tag[NTAG-1].blk + 1) << LB) > rows) ? rows : ((tag[NTAG-1].blk + 1) << LB);
      case (state)
        IDLE: if (start) begin
          y_rdy <= '0;
          if (cfg_rows == 0 || cfg_groups == 0) done <= 1'b1;
          else begin
            state <= RUN; g <= '0; blk <= '0; addr <= cfg_base;
            groups <= cfg_groups; rows <= cfg_rows; nblk <= (cfg_rows + LANES - 1) >> LB;
          end
        end
        RUN: begin
          addr <= addr + 1;
          if (g == groups - 1) begin
            g <= '0;
            if (blk == nblk - 1) state <= DRAIN;
            else blk <= blk + 1;
          end else g <= g + 1;
        end
        DRAIN: if (!pipe_busy) begin state <= IDLE; done <= 1'b1; end
        default: state <= IDLE;
      endcase
    end
  end

  logic [XW-1:0] rd_x;
  always_ff @(posedge clk) if (issue) rd_x <= xmem[g[XA-1:0]];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) for (int k = 0; k < NTAG; k++) tag[k] <= '0;
    else begin
      tag[0].valid <= issue;
      tag[0].first <= (g == 0);
      tag[0].last  <= (g == groups - 1);
      tag[0].blk   <= blk;
      for (int k = 1; k < NTAG; k++) tag[k] <= tag[k-1];
    end
  end

  function automatic logic signed [PW-1:0] product(input logic [WBITS-1:0] w, input logic [ABITS-1:0] a);
    logic signed [ABITS:0] xs;
    xs = {a[ABITS-1], a};
    if (TERNARY) product = (w[1:0] == 2'b01) ? PW'(xs) : (w[1:0] == 2'b11) ? PW'(-xs) : '0;
    else         product = PW'($signed(w) * $signed(a));
  endfunction

  logic [DBITS-1:0] d1, d2, d3;
  always_ff @(posedge clk) begin
    d1 <= rd_x[GROUP*ABITS +: DBITS];
    d2 <= d1;
    d3 <= d2;
  end

  logic signed [ACCW-1:0] nxt [LANES];
  for (genvar l = 0; l < LANES; l++) begin : g_lane
    logic [WW-1:0]            rd_w;
    logic signed [W1-1:0]     s1 [N1];
    logic [SBITS-1:0]         sc1, sc2;
    logic signed [DW-1:0]     s2;
    logic signed [DW+SBITS:0] m3;
    logic signed [ACCW-1:0]   sh4, acc;

    always_ff @(posedge clk) if (issue) rd_w <= wmem[l][addr[WA-1:0]];

    always_ff @(posedge clk) if (tag[0].valid) begin
      for (int k = 0; k < N1; k++) begin
        logic signed [W1-1:0] t [4];
        for (int i = 0; i < 4; i++) t[i] = W1'(product(rd_w[(4*k+i)*WBITS +: WBITS], rd_x[(4*k+i)*ABITS +: ABITS]));
        s1[k] <= (t[0] + t[1]) + (t[2] + t[3]);
      end
      sc1 <= rd_w[GROUP*WBITS +: SBITS];
    end

    always_ff @(posedge clk) if (tag[1].valid) begin
      logic signed [DW-1:0] t [N1];
      for (int k = 0; k < N1; k++) t[k] = DW'(s1[k]);
      for (int w = N1 / 2; w >= 1; w = w / 2)
        for (int k = 0; k < w; k++) t[k] = t[2*k] + t[2*k+1];
      s2 <= t[0];
      sc2 <= sc1;
    end

    always_ff @(posedge clk) if (tag[2].valid) m3 <= s2 * $signed({1'b0, sc2});
    always_ff @(posedge clk) if (tag[3].valid) sh4 <= ACCW'(m3) <<< d3;
    assign nxt[l] = (tag[NTAG-1].first ? '0 : acc) + sh4;
    always_ff @(posedge clk) if (tag[NTAG-1].valid) acc <= nxt[l];
  end

  always_ff @(posedge clk)
    if (tag[NTAG-1].valid && tag[NTAG-1].last)
      for (int l = 0; l < LANES; l++) begin
        logic [31:0] row;
        row = (tag[NTAG-1].blk << LB) + 32'(l);
        if (row < rows) ymem[row[YA-1:0]] <= nxt[l];
      end

endmodule
