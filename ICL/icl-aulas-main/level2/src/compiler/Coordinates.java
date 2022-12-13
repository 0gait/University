package compiler;

import util.Pair;

/**
 * Represents coordinates in the compiler environment.
 */
public class Coordinates extends Pair<Integer, String> {

    public Coordinates(Integer frameId, String fieldId) {
        super(frameId, fieldId);
    }

    @Override
    public String toString() {
        return String.format("Coordinates [frameId=%d, fieldId=%s]", getLeft(), getRight());
    }
}
