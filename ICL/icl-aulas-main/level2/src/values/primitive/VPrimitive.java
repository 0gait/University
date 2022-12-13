package values.primitive;

import values.IValue;

/**
 * Represents a primitive value.
 */
abstract class VPrimitive<E> implements IValue
{
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
        return this.value.toString();
    }
}
