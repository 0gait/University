package typeError;

/**
 * Represents an error of data types when trying to perform an operation.
 */
public class TypeErrorException extends RuntimeException
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
    
}
