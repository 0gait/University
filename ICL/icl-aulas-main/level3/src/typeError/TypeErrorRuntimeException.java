package typeError;

/**
 * Represents an error of data types when trying to perform an operation at runtime (interpreter).
 */
public class TypeErrorRuntimeException extends RuntimeException
{
    protected TypeErrorException typeErrorException;

    /**
     * @param message
     */
    public TypeErrorRuntimeException(String message) {
        super(message,new TypeErrorException(message));
        this.typeErrorException = (TypeErrorException)this.getCause();
    }

    /**
     * @param cause
     */
    public TypeErrorRuntimeException(TypeErrorException cause) {
        super(cause.getMessage(),cause);
        this.typeErrorException = cause;
    }

    /**
     * @param message
     * @param cause
     */
    public TypeErrorRuntimeException(String message, TypeErrorException cause) {
        super(message, cause);
        this.typeErrorException = cause;
    }

    /**
     * Gets the cause of this Type Error.
     * @return the typeErrorException
     */
    public TypeErrorException getTypeErrorException() {
        return typeErrorException;
    }



    
    
}
