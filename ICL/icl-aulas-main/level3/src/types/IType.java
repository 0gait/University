package types;


public interface IType
{
    /**
     * A string representation of this type.
     * @return
     */
    String show();
    
    /**
     * Get a String representation of the type using the JVM syntax.
     * @return
     */
    String getJvmType();

    String getSimpleName();

}
