package types.primitves;

import types.IType;

public abstract class TypePrimitive implements IType
{
    protected Class<?> classType;

    /**
     * @param classType
     */
    public TypePrimitive(Class<?> classType) {
        this.classType = classType;
    }

    @Override
    public String show() {
        return this.classType.getSimpleName();
    }

    @Override
    public boolean equals(Object obj) {
        
        // There is no need to override this method because there is only one instance of
        // each primitive type in the program.

        return super.equals(obj);
    }

    @Override
    public String toString(){
        return show();
    }
}
