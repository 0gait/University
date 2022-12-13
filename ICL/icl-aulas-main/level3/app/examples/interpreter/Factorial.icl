def
   fact= fun x -> if x == 0 then 1 else x*fact(x-1) end end
in
  fact(5)
end;;