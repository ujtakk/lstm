`include "parameters.svh"

module top
  ( input   clk
  , input   xrst
  , input   req
  , output  ack
  , output  img_we
  , output  img_addr
  , output  img_wdata
  );

  ctrl ctrl(.*);

  for (genvar i = 0; i < CORE; i++) begin : pe
    mem_sp #(DWIDTN, NETSIZE) mem_set(
      .*
    );

    core core(
      .*
    );
  end : pe

endmodule
