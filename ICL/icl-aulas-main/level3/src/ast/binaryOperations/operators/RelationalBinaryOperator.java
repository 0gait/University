package ast.binaryOperations.operators;

public enum RelationalBinaryOperator
{
    /**
     * Represents an equals comparison of 2 objects.
     */
    EQUALS("=="),

    /**
     * Represents a a not equals comparison of 2 objects.
     */
    NOT_EQUALS("!="),

    /**
     * Represents a greater than comparison of 2 objects.
     */
    GREATER_THAN(">"),

    /**
     * Represents a greater than or equal to comparison of 2 objects.
     */
    GREATER_THAN_OR_EQUAL_TO(">="),

    /**
     * Represents a less than comparison of 2 objects.
     */
    LESS_THAN("<"),

    /**
     * Represents a less than or equal to comparison of 2 objects.
     */
    LESS_THAN_OR_EQUAL_TO("<=");


    private String operator;

    private RelationalBinaryOperator(String operator)
    {
        this.operator = operator;
    }

    public String getOperator()
    {
        return this.operator;
    }

    public static RelationalBinaryOperator parseOperator(String operator)
    {
        for (RelationalBinaryOperator op : RelationalBinaryOperator.values())
        {
            if (op.getOperator().equals(operator))
                return op;  
        }

        throw new IllegalArgumentException();
    }
}
