package types;

public interface IType
{
    String show();
    
    /**
     * Get a String representation of the type using the JVM syntax.
     * @return
     */
    String getJvmType();
}
