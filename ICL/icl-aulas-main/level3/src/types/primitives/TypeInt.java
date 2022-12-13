package types.primitives;

public class TypeInt extends TypePrimitive implements TypeJavaPrimitive
{
    public static final String TYPE_NAME = "Integer";
    private static final String JVM_TYPE = "I";

    /**
     * @param classType
     */
    private TypeInt() {
        super(Integer.class);
    }

    public static final TypeInt TYPE = new TypeInt();

    @Override
    public int hashCode() {
        return 1;        
    }

    @Override
    public String getJvmType() {
        return JVM_TYPE;
    }
}
