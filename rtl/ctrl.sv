`include "parameters.svh"

module ctrl
  ( input   clk
  , input   xrst
  , input   req
  , output  ack
  );

  ctrl_bus bus_core();

  ctrl_core ctrl_core(
    .in_ctrl  (bus_*),
    .out_ctrl (bus_core),
    .*
  );

endmodule
