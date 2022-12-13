package types;

/**
 * Represents a reference type.
 */
public class TypeRef implements IType
{
    public static final String TYPE_NAME = "Ref";

    protected IType valueType;

    /**
     * @param valueType
     */
    public TypeRef(IType valueType) {
        this.valueType = valueType;
    }

    @Override
    public String show() {
        return TYPE_NAME + " ( " + valueType.show() + " ) ";
    }

    /**
     * The type this reference points to.
     * @return
     */
    public IType getValueType()
    {
        return this.valueType;
    }

    /**
     * Gets the inner type of this reference, i.e.,
     * if {@link #getValueType()} is a reference then calls
     * {@link #getInnerType()} in it, otherwise returns the type of {@link #getValueType()}.
     * @return The inner type of this reference
     */
    public IType getInnerType()
    {
        IType type = getValueType();
        while (type instanceof TypeRef)
        {
            type = ((TypeRef)type).getValueType();
        }

        return type;
    }

    @Override
    public boolean equals(Object obj)
    {
        boolean checked = super.equals(obj);

        if (checked)
            return true;
        
        checked = obj instanceof TypeRef;

        // if it is not a reference
        if (!checked)
            return false;

        TypeRef otherRef = (TypeRef)obj;

        // check if the types of the values in the references are equal.
        return getValueType().equals(otherRef.getValueType());
    }

    @Override
    public String getSimpleName(){
       return "ref_of_" + valueType.getSimpleName();
    }

    @Override
    public String toString(){
        return show();
    }

    @Override
    public int hashCode(){
        String s = toString();
        int res = 0;
        int n = s.length();
       for(int i = 0; i < s.length(); i++){
        res += s.charAt(i)*31^(n-(i + 1));

       }
       return res;
    }

    @Override
    public String getJvmType() {
        return "L" + getSimpleName() + ";";
    }

}
