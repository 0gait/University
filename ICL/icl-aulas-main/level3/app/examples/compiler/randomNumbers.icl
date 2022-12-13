def
    mod = fun dividend:int, divisor:int ->
        dividend - divisor * (dividend / divisor)
    end
    seed = new 20
    random =
        fun -> seed := mod(8121 * !seed + 28411, 181); !seed
    end

    isInsideCircle =
        fun x:int, y:int -> (x * x) + (y * y) <= 32767
    end
in
    def
        i = new 0
        s = new 0
    in
        while !i < 30000 do
            def
                x  = random()
                y  = random()
            in
                if (isInsideCircle(x, y)) then s := !s + 1
                else !s
                end;
                i := !i + 1
            end
        end;
        println (4 * !s * 100 / 30000)
    end
end;;