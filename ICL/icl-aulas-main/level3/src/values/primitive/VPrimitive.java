package values.primitive;

import types.IType;
import types.primitives.TypePrimitive;
import values.IValue;

/**
 * Represents a primitive value.
 */
public abstract class VPrimitive<E extends Comparable<E>> implements IValue, Comparable<VPrimitive<E>>
{
    public static final String TYPE_NAME = "Primitive";

    private E value;

    /**
     * @param value
     */
    protected VPrimitive(E value)
    {
        this.value = value;
    }

    /**
     * @return the value
     */
    public E getValue()
    {
        return this.value;
    }

    @Override
    public String show()
    {
        return getValue().toString();
    }
    
    @Override
    public String toString() {
        return show();
    }

    /**
     * Gets the primitive type of this primitive value.
     * @return the primitive type of this primitive value.
     */
    public abstract TypePrimitive getPrimitiveType();

    @Override
    public IType getType() {
        return getPrimitiveType();
    }

    @Override
    public int compareTo(VPrimitive<E> o)
    {
        return getValue().compareTo(o.getValue());
    }

    @Override
    public boolean equals(Object obj) {
        
        if (this == obj)
            return true;

        if (!(obj instanceof VPrimitive))
            return false;

        VPrimitive<?> other = (VPrimitive<?>)obj;
        
        return getValue().equals(other.getValue());
    }
}
