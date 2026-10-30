module reduce #(
  parameter int N   = 32,
  parameter int W   = 48,
  parameter int OP  = 0,
  parameter int IW  = 1,
  parameter int LPS = 2
) (
  input  logic          clk,
  input  logic          iv,
  input  logic [W-1:0]  din [N],
  input  logic [IW-1:0] iin [N],
  output logic          ov,
  output logic [W-1:0]  dout,
  output logic [IW-1:0] iout
);

  localparam int LV = $clog2(N);
  localparam int ST = (LV + LPS - 1) / LPS;

  function automatic logic right(input logic [W-1:0] a, input logic [W-1:0] b);
    right = (OP == 1) ? (b > a) : ($signed(b) > $signed(a));
  endfunction

  if (ST == 0) begin : g_wire
    assign ov = iv;
    assign dout = din[0];
    assign iout = iin[0];
  end else begin : g_tree
    for (genvar s = 0; s < ST; s++) begin : g_st
      localparam int NI = N >> (s * LPS);
      localparam int L  = (LV - s * LPS < LPS) ? (LV - s * LPS) : LPS;
      localparam int NO = NI >> L;
      logic          vi;
      logic [W-1:0]  xi [NI];
      logic [IW-1:0] ii [NI];
      logic          vo;
      logic [W-1:0]  xo [NO];
      logic [IW-1:0] io [NO];
      if (s == 0) begin : g_in
        assign vi = iv;
        for (genvar k = 0; k < NI; k++) begin : g_k
          assign xi[k] = din[k];
          assign ii[k] = iin[k];
        end
      end else begin : g_in
        assign vi = g_st[s-1].vo;
        for (genvar k = 0; k < NI; k++) begin : g_k
          assign xi[k] = g_st[s-1].xo[k];
          assign ii[k] = g_st[s-1].io[k];
        end
      end
      always_ff @(posedge clk) begin
        vo <= vi;
        if (vi) begin
          logic [W-1:0]  t  [NI];
          logic [IW-1:0] ti [NI];
          for (int k = 0; k < NI; k++) begin t[k] = xi[k]; ti[k] = ii[k]; end
          for (int lv = 0; lv < L; lv++)
            for (int k = 0; k < (NI >> (lv + 1)); k++)
              if (OP == 0) t[k] = t[2*k] + t[2*k+1];
              else if (right(t[2*k], t[2*k+1])) begin t[k] = t[2*k+1]; ti[k] = ti[2*k+1]; end
              else begin t[k] = t[2*k]; ti[k] = ti[2*k]; end
          for (int k = 0; k < NO; k++) begin xo[k] <= t[k]; io[k] <= ti[k]; end
        end
      end
    end
    assign ov = g_st[ST-1].vo;
    assign dout = g_st[ST-1].xo[0];
    assign iout = g_st[ST-1].io[0];
  end

endmodule

module mreduce #(
  parameter int M   = 4,
  parameter int N   = 32,
  parameter int W   = 48,
  parameter int OP  = 0,
  parameter int LPS = 2
) (
  input  logic         clk,
  input  logic         iv,
  input  logic [W-1:0] din [M][N],
  output logic         ov,
  output logic [W-1:0] dout [M]
);

  localparam int LV = $clog2(N);
  localparam int ST = (LV + LPS - 1) / LPS;

  function automatic logic [W-1:0] comb(input logic [W-1:0] a, input logic [W-1:0] b);
    if (OP == 0) comb = a + b;
    else if (OP == 1) comb = (b > a) ? b : a;
    else comb = ($signed(b) > $signed(a)) ? b : a;
  endfunction

  if (ST == 0) begin : g_wire
    assign ov = iv;
    for (genvar m = 0; m < M; m++) begin : g_m
      assign dout[m] = din[m][0];
    end
  end else begin : g_tree
    for (genvar s = 0; s < ST; s++) begin : g_st
      localparam int NI = N >> (s * LPS);
      localparam int L  = (LV - s * LPS < LPS) ? (LV - s * LPS) : LPS;
      localparam int NO = NI >> L;
      logic         vo;
      logic [W-1:0] xo [M][NO];
      always_ff @(posedge clk) begin
        logic vi;
        vi = (s == 0) ? iv : g_st[(s == 0) ? 0 : s - 1].vo;
        vo <= vi;
        if (vi)
          for (int m = 0; m < M; m++) begin
            logic [W-1:0] t [NI];
            for (int k = 0; k < NI; k++) t[k] = (s == 0) ? din[m][k] : g_st[(s == 0) ? 0 : s - 1].xo[m][k];
            for (int lv = 0; lv < L; lv++)
              for (int k = 0; k < (NI >> (lv + 1)); k++) t[k] = comb(t[2*k], t[2*k+1]);
            for (int k = 0; k < NO; k++) xo[m][k] <= t[k];
          end
      end
    end
    assign ov = g_st[ST-1].vo;
    for (genvar m = 0; m < M; m++) begin : g_o
      assign dout[m] = g_st[ST-1].xo[m][0];
    end
  end

endmodule
