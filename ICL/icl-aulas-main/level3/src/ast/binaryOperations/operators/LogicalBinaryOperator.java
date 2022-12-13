package ast.binaryOperations.operators;

public enum LogicalBinaryOperator
{
    /**
     * Represents a logical AND of 2 boolean values.
     */
    AND("&&"),

    /**
     * Represents a logical OR of 2 boolean values.
     */
    OR("||");


    private String operator;

    private LogicalBinaryOperator(String operator)
    {
        this.operator = operator;
    }

    public String getOperator()
    {
        return this.operator;
    }

    public static LogicalBinaryOperator parseOperator(String operator)
    {
        for (LogicalBinaryOperator op : LogicalBinaryOperator.values())
        {
            if (op.getOperator().equals(operator))
                return op;  
        }

        throw new IllegalArgumentException();
    }
}
