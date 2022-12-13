# Projeto ICL

Aulas práticas de Interpretação e Compilação de Linguagens.

O projeto foi desenvolvido em diferentes fases, sendo que a versão mais completa é a linguagem de nível 3, disponível na pasta [level3](https://github.com/henriquej-0904/icl-aulas/tree/main/level3).

Trabalho realizado por:

- Henrique Campos Ferreira - 55065
- José Dias - 55015


## Intruções para correr o programa

O programa foi desenvolvido com base no docker, pois permitiu instalarmos todas as ferramentas que precisámos, ex: jasmin, javacc, variáveis de ambiente, etc.

- Compilar o programa ([build.sh](https://github.com/henriquej-0904/icl-aulas/blob/main/level3/build.sh))
- Executar o container com o programa compilado ([run.sh](https://github.com/henriquej-0904/icl-aulas/blob/main/level3/run.sh))

Para correr o programa em modo debug (apresentar os ficheiros jasmin gerados, stack trace dos erros, etc) deve-se executar o script [debug.sh](https://github.com/henriquej-0904/icl-aulas/blob/main/level3/debug.sh). Os ficheiros jasmin gerados a partir de uma compilação estão acessíveis numa pasta [app](https://github.com/henriquej-0904/icl-aulas/tree/main/level3/app)/jFiles.

Para se obter informação sobre os comandos que o programa aceita deve-se executar:
```bash
java -jar ICL.jar

```
  
## Correr o programa em modo interpretador

Para correr o programa em modo interpretador pode-se utilizar a shell ou fornecer um ficheiro com um programa. Existem vários exemplos de programas para o interpretador que estão disponíveis em [examples/interpreter](https://github.com/henriquej-0904/icl-aulas/tree/main/level3/app/examples/interpreter).
  
Exemplo:
```bash
java -jar ICL.jar -e examples/interpreter/PrintRecords.icl

```
## Correr o programa em modo compilador
  
Neste modo, deve ser utilizada a flag -c que indica que se pretende compilar o programa fornecido. Os exemplos para o compilador estão disponíveis em [examples/compiler](https://github.com/henriquej-0904/icl-aulas/tree/main/level3/app/examples/compiler).

Exemplo:
```bash
java -jar ICL.jar -c examples/compiler/PrintRecords.icl

```

É obrigatório fornecer os tipos dos argumentos de funções (recursivas ou não), mas não é necessário declarar os tipos dos binds numa expressão DEF. Caso sejam declarados os tipos dos binds, o typechecker verifica se o tipo declarado corresponde ao tipo inferido. No exemplo que se segue, são criados 3 binds "func", "x" e "y" em que não são declarados os tipos e um bind "z" em que é verificado se o tipo declarado (bool) corresponde ao tipo inferido (tipo de 'true' -> bool).


```bash
def
  func = fun x:int, y:int -> x+y end
  x = 1
  y = "ola"
  z:bool = true
in
  println (func(1,2))
end;;

```
  
Ao contrário do interpretador, é necessário indicar se uma função é recursiva, sendo utilizada a keyword **rec**. Além disso, é necessário fornecer o tipo da função, por exemplo:
  
```bash
def
  rec fact:(int)int = fun x:int -> if x == 0 then 1 else x*fact(x-1) end end
in
  println (fact(5))
end;;

```

Para criar Records deve ser utilizada a seguinte sintaxe: [ id_1:tipo_1 = valor_1, id_1:tipo_1 = valor_1 ]. Tal como nos DEF, o typechecker funciona da mesma forma, ou seja, não é obrigatório fornecer os tipos. Um exemplo de utilização de Records é o seguinte:

```bash
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

```

## Otimizações do compilador

O programa otimiza a compilação de expressões booleanas quando estas são utilizadas em contruções do tipo IF ou WHILE (Short Circuit). Além disso, as expressões booleanas são otimizadas da seguinte forma:

- x || y -> Só se avalia y no caso de x ser falso.
- x && y -> Só se avalia y no caso de x ser verdadeiro.

## Otimizações do Environment
O programa optimiza os environments fazendo com que cada um guarde uma referência para o anterior.

## Funções de biblioteca

As funções de biblioteca, tal como todas as outras, necessitam de parênteses "(...)" para se passar os argumentos.

- println -> Função que recebe um tipo primitivo e imprime a sua representação em texto.
- length -> Devolve o tamanho de uma string.
- printf->  Função que imprime uma string formatada baseado numa string e os argumentos especificados.
- isEmpty-> Indica se uma string é vazia.

## Tipos da linguagem
De seguida, apresentam-se os tipos definidos na linguagem. Cada tipo, à esquerda, está associado a uma sintaxe específica que o representa, à direita.

- Integer -> int
- Boolean -> bool
- String -> string
- Reference -> ref_tipo, sendo 'tipo' o tipo que a referência está associada. Por exemplo, ref_int representa uma referência para inteiros.
- Funções -> (tipo_arg1 , tipo_arg2 , ...)tipo_resultado
- Records -> record(id_1 : tipo_1 , id_2 : tipo_2 , ...), em que os id's correspondem aos nomes dos campos de um Record.
- Void -> void, só pode ser dado para retorno de uma função.
  
  Os tipos primitivos da linguagem são inteiros, booleanos e strings e por isso pode-se usar todos os operadores de comparação ( ==, !=, >, >=, <, <=) sobre estes tipos.
  Os operadores aritméticos ( +, -, \*, / ) estão definidos para o tipo Integer.
  A concatenação de strings está definida através do operador da soma ( + ).
  
  
  
  
  
  
  
  
  
  
  
  
