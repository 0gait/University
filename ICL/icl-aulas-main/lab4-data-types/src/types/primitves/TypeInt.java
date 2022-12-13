package types.primitves;

public class TypeInt extends TypePrimitive
{
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
        // TODO Auto-generated method stub
        return 1;
        
    }

    @Override
    public String getJvmType() {
        return JVM_TYPE;
    }
}
