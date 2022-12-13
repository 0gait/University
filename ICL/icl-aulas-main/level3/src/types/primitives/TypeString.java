package types.primitives;

public class TypeString extends TypePrimitive
{
    public static final String TYPE_NAME = "String";
    private static final String JVM_TYPE = "Ljava/lang/String;";

    public static final TypeString TYPE = new TypeString();

    private TypeString() {
        super(String.class);
    }

    @Override
    public String getJvmType() {
        return JVM_TYPE;
    }

    @Override
    public int hashCode() {
        return 3;
    }
    
}
