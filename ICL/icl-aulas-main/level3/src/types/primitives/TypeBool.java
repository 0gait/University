package types.primitives;

public class TypeBool extends TypePrimitive implements TypeJavaPrimitive
{
    public static final String TYPE_NAME = "Boolean";
    private static final String JVM_TYPE = "Z";

    /**
     * @param classType
     */
    private TypeBool() {
        super(Boolean.class);
    }

    public static final TypeBool TYPE = new TypeBool();

    @Override
    public int hashCode() {
        return 2;
    }

    @Override
    public String getJvmType() {
        return JVM_TYPE;
    }
}
