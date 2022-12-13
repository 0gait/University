package ast.binaryOperations.operators;

public enum ArithmeticBinaryOperator
{
    /**
     * Represents an addition of 2 objects.
     */
    ADD("+"),

    /**
     * Represents a subtraction of 2 objects.
     */
    SUB("-"),

    /**
     * Represents a multiplication of 2 objects.
     */
    MUL("*"),

    /**
     * Represents a division of 2 objects.
     */
    DIV("/");

    
    private String operator;

    private ArithmeticBinaryOperator(String operator)
    {
        this.operator = operator;
    }

    public String getOperator()
    {
        return this.operator;
    }

}
