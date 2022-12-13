def 
    createCounter = fun x:int -> 
        def x = new x in
            [
                inc = fun -> x := !x + 1 end,
                add = fun y:int -> x := !x + y end,
                sub = fun y:int -> x := !x - y end,
                get = fun -> !x end
            ]
        end
    end
    
in
    def
        primes = fun range1:int, range2:int -> 
            def
                mod = fun dividend:int, divisor:int ->
                    dividend - divisor * (dividend / divisor) end
                
                counter = if( range1 <= 2 ) then createCounter(2) 
                            else createCounter(range1) end
                checkPrime = fun toCheck:int,curr:int-> 
                    if( toCheck == 2 || toCheck == 3) then
                        true
                    else (
                        def current = createCounter(curr) 
                            isPrime = new (true) in  
                            while(!isPrime && current.get() >= 3) do (
                                   isPrime :=  mod(toCheck,current.get()) != 0;
                                   current.sub(2)
                            )
                            end;
                            !isPrime
                        end
                    )
                       
                    end
                end

                finish = fun -> counter.get() > range2 end 

                getFirstPrime = fun -> if(counter.get() == 2) then 2
                                        else (
                                            if(mod(counter.get(),2) == 0) then
                                                counter.inc()
                                            end;
                                            def found = new (false) in 
                                                while(~(finish()) && ~(!found )) do (
                                                    found := checkPrime(counter.get(), counter.get() - 2);
                                                    if(~(!found )) then
                                                        counter.add(2)
                                                    end
                                                )
                                                end;
                                                counter.get()
                                            end
                                        )
                                        end
                    end
                getNextPrime = fun -> 
                         if(counter.get() == 2) then 
                            counter.inc()
                         else(
                                counter.add(2);
                                def found = new (false) in 
                                    while(~(finish()) && ~(!found)) do (
                                        found := checkPrime(counter.get(), counter.get() - 2);
                                        if(~(!found)) then
                                          counter.add(2)
                                        end
                                    )
                                    end;
                                    counter.get()
                                end
                           )
                           end
                    end
            in
                getFirstPrime();
                [
                    hasNext =  fun -> ~ (finish()) end,

                    get = fun -> def toReturn = counter.get() in getNextPrime(); toReturn end end
                ]
            end
        end
    in
        def
            range1 = 30000
            range2 = 32000
            primes = primes(range1, range2)
        in
            printf("Primes between %d and %d:\n",range1, range2);

            while (primes.hasNext()) do
                println (primes.get())
            end;

            println("Finished!")
        end
    end
end;;