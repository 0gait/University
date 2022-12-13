def
  rec fact:(int)int = fun x:int -> if x == 0 then 1 else x*fact(x-1) end end
in
  println (fact(5))
end;;