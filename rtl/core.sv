`include "parameters.svh"

module core
  ( input                       clk
  , input                       xrst
  , output signed [DWIDTH-1:0]  result
  );

  mac full_upward(
    .*
  );

  bias bias_upward(
    .*
  );

  mac full_lateral(
    .*
  );

  extract extract(
    .*
  );

  tanh tanh_a(
    .*
  );

  sigmoid sigmoid_i(
    .*
  );

  sigmoid sigmoid_f(
    .*
  );

  sigmoid sigmoid_o(
    .*
  );

  tanh tanh_cn(
    .*
  );

endmodule
