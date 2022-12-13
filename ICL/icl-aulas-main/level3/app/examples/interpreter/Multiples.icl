def 
    createCounter = fun x -> 
        def x = new x in
            [
                inc = fun -> x := !x + 1 end,
                add = fun y -> x := !x + y end,

                get = fun -> !x end
            ]
        end
    end
    
in
    def
        multiplesOf = fun x, range1, range2-> 
            def
                mod = fun dividend , divisor ->
                    dividend - divisor * (dividend / divisor) end
                
                initCounter = fun ->
                    if range1 <= x then
                        createCounter(x)
                    else
                        def counter = createCounter(range1)
                            found = new (false)
                        in
                            while ( ~(!found) ) do
                                found := counter.inc() > range2 || mod(counter.get(), x) == 0
                            end;
                            counter
                        end
                    end
                end

                counter = initCounter()
            in
                [
                    hasNext = fun -> counter.get() <= range2 end,

                    get = fun -> def toReturn = counter.get() in
                                    counter.add(x);
                                    toReturn
                                end
                          end  
                ]
            end
        end
    in
        def
            x = 7
            range1 = 0
            range2 = 100
            multiples = multiplesOf(x, range1, range2)
        in
            printf("Multiples of %d between %d and %d:\n", x, range1, range2);
            println("");
            while (multiples.hasNext()) do
                println (multiples.get())
            end;
            println("Finished!")
        end
    end
end;;