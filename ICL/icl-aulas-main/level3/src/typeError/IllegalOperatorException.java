package typeError;

/**
 * Represents an error when trying to apply an operator to illegal arguments.
 */
public class IllegalOperatorException extends TypeErrorException
{
    protected final static String DEFAULT_ERROR_MSG_FORMAT = "Illegal arguments to '%s' operator.";

    protected final static String EXPECTED_FORMAT = "Expected '%s' and got '%s'";

    protected final static String ILLEGAL_TYPE_FORMAT = "Illegal arguments to '%s' operator for type '%s'.";

    public final String operator;

    /**
     * @param operator
     */
    public IllegalOperatorException(String operator, String typeExpected, String typeGot)
    {
        super(String.format(DEFAULT_ERROR_MSG_FORMAT + "\n" + EXPECTED_FORMAT, operator, typeExpected, typeGot));
        this.operator = operator;
    }

    /**
     * @param message
     * @param operator
     */
    public IllegalOperatorException(String message, String operator, String typeExpected, String typeGot) {
        super(
            new StringBuilder(String.format(DEFAULT_ERROR_MSG_FORMAT, operator))
                .append('\n')
                .append(message)
                .append('\n')
                .append(String.format(EXPECTED_FORMAT, typeExpected, typeGot))
                .toString()
            );
        this.operator = operator;
    }

    /**
     * @param operator - The operator.
     * @param illegalType - The type for this illegal operation.
     */
    public IllegalOperatorException(String operator, String illegalType)
    {
        super(String.format(ILLEGAL_TYPE_FORMAT, operator, illegalType));
        this.operator = operator;
    }
}
