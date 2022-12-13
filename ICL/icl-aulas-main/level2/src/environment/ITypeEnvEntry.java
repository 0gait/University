package environment;

import types.IType;

/**
 * An entry in the environment for types.
 */
public class ITypeEnvEntry extends EnvironmentEntry<IType> {

    public ITypeEnvEntry(String id, IType value) {
        super(id, value);
    }
    
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(getLeft());
        builder.append('=');
        builder.append(getRight().show());

        return builder.toString();
    }
}
