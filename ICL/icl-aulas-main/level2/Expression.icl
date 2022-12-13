def
  f:(int)int = fun x:int -> if x==0 then 1 else x*f(x-1) end  end
in
  println (f( 3 ) + f(2) + f(4))
end;;