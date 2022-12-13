def
   fib = fun x -> if (x == 0|| x == 1) then 1 else fib(x-1) + fib(x - 2) end end
in
  fib(8)
end;;