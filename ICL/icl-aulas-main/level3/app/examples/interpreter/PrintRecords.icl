def
    printPerson = fun person ->
       printf("Person with name: %s and age: %d", person.name, person.age) end
in
    def
        person1 = [ name = "Henrique", age = 21 ]
        person2 = [ name = "José", age = 21 ]
    in
        printPerson(person1);
        println("");
        printPerson(person2)
    end
end;;
