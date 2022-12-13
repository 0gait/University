def
    printPerson = fun person:record(name:string, age:int)->
       printf("Person with name: %s and age: %d\n", person.name, person.age) end
in
    def
        person1 = [ name = "Henrique", age = 21 ]
        person2 = [ name = "José", age = 21 ]
    in
        printPerson(person1);
        printPerson(person2)
    end
end;;
