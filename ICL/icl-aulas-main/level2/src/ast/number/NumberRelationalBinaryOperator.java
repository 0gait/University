package ast.number;

public enum NumberRelationalBinaryOperator
{
    /**
     * Represents an equals comparison of 2 numbers.
     */
    EQUALS("=="),

    /**
     * Represents a a not equals comparison of 2 numbers.
     */
    NOT_EQUALS("!="),

    /**
     * Represents a greater than comparison of 2 numbers.
     */
    GREATER_THAN(">"),

    /**
     * Represents a greater than or equal to comparison of 2 numbers.
     */
    GREATER_THAN_OR_EQUAL_TO(">="),

    /**
     * Represents a less than comparison of 2 numbers.
     */
    LESS_THAN("<"),

    /**
     * Represents a less than or equal to comparison of 2 numbers.
     */
    LESS_THAN_OR_EQUAL_TO("<=");


    private String operator;

    private NumberRelationalBinaryOperator(String operator)
    {
        this.operator = operator;
    }

    public String getOperator()
    {
        return this.operator;
    }

    public static NumberRelationalBinaryOperator parseOperator(String operator)
    {
        for (NumberRelationalBinaryOperator op : NumberRelationalBinaryOperator.values())
        {
            if (op.getOperator().equals(operator))
                return op;  
        }

        throw new IllegalArgumentException();
    }
}
