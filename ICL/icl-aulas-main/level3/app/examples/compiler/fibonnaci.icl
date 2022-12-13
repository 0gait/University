def
  rec fib:(int)int = fun x:int -> if (x == 0|| x == 1) then 1 else fib(x-1) + fib(x - 2) end end
in
  println (fib(5))
end;;