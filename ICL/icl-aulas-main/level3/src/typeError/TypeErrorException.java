package typeError;

/**
 * Represents an error of data types when trying to perform an operation at compile time.
 */
public class TypeErrorException extends Exception
{
    /**
     * @param message
     */
    public TypeErrorException(String message) {
        super(message);
    }

    /**
     * @param cause
     */
    public TypeErrorException(Throwable cause) {
        super(cause);
    }

    /**
     * @param message
     * @param cause
     */
    public TypeErrorException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * @param message
     * @param cause
     * @param enableSuppression
     * @param writableStackTrace
     */
    public TypeErrorException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }

    /**
     * Converts this Checked TypeErrorException to a Unchecked TypeErrorRuntimeException.
     * @return a new TypeErrorRuntimeException with the same properties.
     */
    public TypeErrorRuntimeException toRuntimeException()
    {
        return new TypeErrorRuntimeException(this);
    }
    
}
