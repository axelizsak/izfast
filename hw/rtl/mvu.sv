module mvu #(
  parameter int LANES   = 8,
  parameter int GROUP   = 32,
  parameter int WBITS   = 4,
  parameter int HBITS   = WBITS,
  parameter int SBITS   = 16,
  parameter int NWW     = (WBITS == 2) ? 56 : 32 * WBITS + SBITS,
  parameter int RW      = (HBITS != WBITS && 32 * HBITS + SBITS > NWW) ? 32 * HBITS + SBITS : NWW,
  parameter int SDEPTH  = 1,
  parameter int ABITS   = 8,
  parameter int DBITS   = 5,
  parameter int DMAX    = 24,
  parameter int WDEPTH  = 16384,
  parameter int HDEPTH  = 1,
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
  input  logic [RW-1:0]                         w_data,
  input  logic                                  s_we,
  input  logic [$clog2(LANES)-1:0]              s_lane,
  input  logic [31:0]                           s_addr,
  input  logic [SBITS-1:0]                      s_data,
  input  logic                                  x_we,
  input  logic [XPORTS-1:0]                     x_mask,
  input  logic [31:0]                           x_addr,
  input  logic [GROUP*ABITS+DBITS-1:0]          x_data [XPORTS],
  input  logic [31:0]                           cfg_base,
  input  logic [31:0]                           cfg_rows,
  input  logic [31:0]                           cfg_groups,
  input  logic                                  cfg_wide,
  input  logic [31:0]                           cfg_sbase,
  input  logic                                  start,
  output logic                                  busy,
  output logic                                  done,
  output logic [31:0]                           y_ready,
  input  logic [31:0]                           y_addr,
  input  logic                                  y_re,
  output logic signed [ACCW-1:0]                y_vec [YPORTS]
);

  localparam bit MIXED  = (HBITS != WBITS);
  localparam bit TPACK  = (WBITS == 2);
  localparam int SA     = SDEPTH > 1 ? $clog2(SDEPTH) : 1;
  localparam int MB     = HBITS > WBITS ? HBITS : WBITS;
  localparam int WW     = GROUP * MB + SBITS;
  localparam int XW     = GROUP * ABITS + DBITS;
  localparam int LEVELS = $clog2(GROUP);
  localparam int PW     = MB + ABITS;
  localparam int DW     = PW + LEVELS;
  localparam int LB     = $clog2(LANES);
  localparam int WA     = $clog2(WDEPTH);
  localparam int HA     = HDEPTH > 1 ? $clog2(HDEPTH) : 1;
  localparam int HWW    = 32 * HBITS + SBITS;
  localparam int XA     = $clog2(XDEPTH);
  localparam int YA     = $clog2(YDEPTH);

  logic [XW-1:0]          xmem [XDEPTH];
  logic signed [ACCW-1:0] ymem [YDEPTH];

  always_ff @(posedge clk) begin
    if (x_we)
      for (int k = 0; k < XPORTS; k++)
        if (x_mask[k]) xmem[XA'(x_addr + 32'(k))] <= x_data[k];
  end

  always_ff @(posedge clk)
    if (y_re) for (int u = 0; u < YPORTS; u++) y_vec[u] <= ymem[YA'(y_addr + 32'(u))];

  typedef enum logic [1:0] {IDLE, RUN, DRAIN} state_t;
  state_t state;
  logic [31:0] g, blk, nblk, groups, rows, addr, y_rdy, sbase;
  logic        wide;
  assign y_ready = start ? '0 : y_rdy;

  localparam int L1     = 2;
  localparam int N1     = GROUP >> L1;
  localparam int W1     = PW + L1;
  localparam int NTAG   = 6;

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
      state <= IDLE; g <= '0; blk <= '0; nblk <= '0; groups <= '0; rows <= '0; addr <= '0; done <= 1'b0; y_rdy <= '0; wide <= 1'b0;
    end else begin
      done <= 1'b0;
      if (tag[NTAG-1].valid && tag[NTAG-1].last)
        y_rdy <= (((tag[NTAG-1].blk + 1) << LB) > rows) ? rows : ((tag[NTAG-1].blk + 1) << LB);
      case (state)
        IDLE: if (start) begin
          y_rdy <= '0;
          wide <= MIXED && cfg_wide;
          sbase <= cfg_sbase;
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

  logic [XW-1:0] rx0, rd_x;
  always_ff @(posedge clk) if (issue) rx0 <= xmem[g[XA-1:0]];
  always_ff @(posedge clk) if (tag[0].valid) rd_x <= rx0;

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

  localparam logic [2559:0] TRITS = 2560'h00000000000000000000000000000000155551575455054d5d5715f5154451d41501435354c53d755d1775c5705cd7d5f17f4551445d11441134751c47d05411074050040d0d4310f4d5344dd314c1334f53c4fdd5751d7745d074ddd771df715c471dc1701c3735cc73df57d1f77c5f07cdfd7f1ff1545415c51140531745c17c45110471044010c4d1304f1d4741dc711c0731f47c1fc15050170441004c1d0701f0140401c01000030340c03c350d0370c4300cc3d0f03f354d435cd1340d3374dc37cc5310c7304c030ccd330cf3d4f43dcf13c0f33f4fc3ff55d5357d4750d4f5dd735fd1744d1f41d0343d374cd3f75dd377dc770dcf7ddf37fc5714c5f11c4313c771cc7f05c1307c0700c0f0dc330fcd734cdf31cc333cf73ccffd5f53d7f47d0f4fddf73dff17c4f1fc1f03c3f37ccf3ff5fd3f7fc7f0fcffdff3ff;

  function automatic logic [GROUP*2-1:0] untrit(input logic [55:0] w);
    logic [69:0] t;
    for (int j = 0; j < 7; j++) t[j*10 +: 10] = TRITS[w[j*8 +: 8]*10 +: 10];
    untrit = t[GROUP*2-1:0];
  endfunction

  function automatic logic signed [PW-1:0] product(input logic [MB-1:0] w, input logic [ABITS-1:0] a, input int bits);
    logic signed [ABITS:0] xs;
    xs = {a[ABITS-1], a};
    if (bits == 2)      product = (w[1:0] == 2'b01) ? PW'(xs) : (w[1:0] == 2'b11) ? PW'(-xs) : '0;
    else if (bits == 4) product = PW'($signed(w[3:0]) * $signed(a));
    else                product = PW'($signed(w[7:0]) * $signed(a));
  endfunction

  logic [DBITS-1:0] d1, d2, d3;
  always_ff @(posedge clk) begin
    d1 <= rd_x[GROUP*ABITS +: DBITS];
    d2 <= d1;
    d3 <= d2;
  end

  logic signed [ACCW-1:0] nxt [LANES];
  for (genvar l = 0; l < LANES; l++) begin : g_lane
    logic [NWW-1:0]           wmem [WDEPTH];
    logic [HWW-1:0]           hmem [HDEPTH];
    logic [SBITS-1:0]         smem [SDEPTH];
    logic [WW-1:0]            rd_w;
    logic [RW-1:0]            raw;
    logic [SBITS-1:0]         rsc;

`ifndef SYNTHESIS
    initial begin
      string f;
      if ($value$plusargs("wmem=%s", f)) $readmemh($sformatf("%s.%0d.hex", f, l), wmem);
      if (MIXED && $value$plusargs("hmem=%s", f)) $readmemh($sformatf("%s.%0d.hex", f, l), hmem);
      if ($value$plusargs("smem=%s", f)) $readmemh($sformatf("%s.%0d.hex", f, l), smem);
    end
`endif

    always_ff @(posedge clk) begin
      if (w_we && w_lane == LB'(l)) begin
        if (MIXED && w_addr[31]) hmem[HA'(w_addr)] <= HWW'(w_data);
        else wmem[w_addr[WA-1:0]] <= NWW'(w_data);
      end
      if (s_we && s_lane == LB'(l)) smem[SA'(s_addr)] <= s_data;
    end
    logic signed [W1-1:0]     s1 [N1];
    logic [SBITS-1:0]         sc1, sc2;
    logic signed [DW-1:0]     s2;
    logic signed [DW+SBITS:0] m3;
    logic signed [ACCW-1:0]   sh4, acc;

    always_ff @(posedge clk) if (issue) begin
      raw <= (MIXED && wide) ? RW'(hmem[HA'(addr)]) : RW'(wmem[addr[WA-1:0]]);
      rsc <= smem[SA'(sbase + blk)];
    end

    always_ff @(posedge clk) if (tag[0].valid)
      rd_w <= (TPACK && !(MIXED && wide)) ? WW'({rsc, untrit(raw[55:0])}) : WW'(raw);

    always_ff @(posedge clk) if (tag[1].valid) begin
      for (int k = 0; k < N1; k++) begin
        logic signed [W1-1:0] t [4];
        for (int i = 0; i < 4; i++)
          t[i] = (MIXED && wide) ? W1'(product(MB'(rd_w[(4*k+i)*HBITS +: HBITS]), rd_x[(4*k+i)*ABITS +: ABITS], HBITS))
                                 : W1'(product(MB'(rd_w[(4*k+i)*WBITS +: WBITS]), rd_x[(4*k+i)*ABITS +: ABITS], WBITS));
        s1[k] <= (t[0] + t[1]) + (t[2] + t[3]);
      end
      sc1 <= (MIXED && wide) ? rd_w[GROUP*HBITS +: SBITS] : rd_w[GROUP*WBITS +: SBITS];
    end

    always_ff @(posedge clk) if (tag[2].valid) begin
      logic signed [DW-1:0] t [N1];
      for (int k = 0; k < N1; k++) t[k] = DW'(s1[k]);
      for (int w = N1 / 2; w >= 1; w = w / 2)
        for (int k = 0; k < w; k++) t[k] = t[2*k] + t[2*k+1];
      s2 <= t[0];
      sc2 <= sc1;
    end

    always_ff @(posedge clk) if (tag[3].valid) m3 <= s2 * $signed({1'b0, sc2});
    always_ff @(posedge clk) if (tag[4].valid) sh4 <= ACCW'(m3) <<< d3;
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
