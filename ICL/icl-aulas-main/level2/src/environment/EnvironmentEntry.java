package environment;

import util.Pair;

/**
 * Represents an entry in the environment.
 */
public class EnvironmentEntry<E> extends Pair<String, E> {

    /**
     * @param id
     * @param value
     */
    public EnvironmentEntry(String id, E value) {
        super(id, value);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(getLeft());
        builder.append('=');
        builder.append(getRight().toString());

        return builder.toString();
    }
}
