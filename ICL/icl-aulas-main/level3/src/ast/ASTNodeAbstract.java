package ast;

import types.IType;
import types.TypeNull;

public abstract class ASTNodeAbstract implements ASTNode
{
    protected IType type;

    protected ASTNodeAbstract()
    {
        this.type = TypeNull.TYPE;
    }

    @Override
    public IType getType() {
        return type;
    }
    
    @Override
    public String toString() {
        return toString(new StringBuilder()).toString();
    }

    /**
     * Converts this AST Node to it's String representation.
     * @param builder
     * @return
     */
    public abstract StringBuilder toString(StringBuilder builder);
}