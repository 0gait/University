package ast.number;

public enum NumberArithmeticBinaryOperator
{
    /**
     * Represents an addition of 2 numbers.
     */
    ADD("+"),

    /**
     * Represents a subtraction of 2 numbers.
     */
    SUB("-"),

    /**
     * Represents a multiplication of 2 numbers.
     */
    MUL("*"),

    /**
     * Represents a division of 2 numbers.
     */
    DIV("/");

    
    private String operator;

    private NumberArithmeticBinaryOperator(String operator)
    {
        this.operator = operator;
    }

    public String getOperator()
    {
        return this.operator;
    }

}
