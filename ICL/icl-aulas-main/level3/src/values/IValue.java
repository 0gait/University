package values;

import types.IType;

/**
 * Represents values.
 */
public interface IValue {
    
    /**
     * Returns a String representation of the value.
     * @return A String representation of the value.
     */
    String show();

    /**
     * Gets the type of this value.
     * @return the type of this value.
     */
    IType getType();

}
