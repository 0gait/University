package types;

public class TypeRef implements IType
{
    public static final String TYPE = "Ref";

    protected IType valueType;

    /**
     * @param valueType
     */
    public TypeRef(IType valueType) {
        this.valueType = valueType;
    }

    @Override
    public String show() {
        return TYPE + " ( " + valueType.show() + " ) ";
    }

    public IType getValueType()
    {
        return this.valueType;
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
    public String toString(){
       return "ref_of_" + valueType;
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
        return "L" + this + ";";
    }

}
