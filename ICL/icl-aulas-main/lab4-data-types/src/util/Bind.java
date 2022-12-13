package util;

import ast.ASTNode;

/**
 * Represents a bind.
 */
public class Bind extends Pair<String, ASTNode>
{

    /**
     * @param id
     * @param node
     */
    public Bind(String id, ASTNode node)
    {
        super(id, node);
    }
}
