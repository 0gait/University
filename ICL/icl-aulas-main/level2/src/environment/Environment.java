package environment;

import java.util.*;
import java.util.function.Consumer;

import util.Utils;

public class Environment<E> implements Cloneable {

	// TODO: Add reference to previous environment instead of calling clone: private Environment<E> previous;

	private Stack<List<EnvironmentEntry<E>>> scopeStack;

	public Environment() {
		scopeStack = new Stack<List<EnvironmentEntry<E>>>();
	}

	private Environment(Stack<List<EnvironmentEntry<E>>> scopeStack) {
		this.scopeStack = scopeStack;
	}

	@SuppressWarnings("unchecked")
	public Environment<E> beginScope() {
		Environment<E> e = (Environment<E>) this.clone();
		e.scopeStack.push(new LinkedList<>());
		return e;
	}

	@SuppressWarnings("unchecked")
	public Environment<E> endScope() {
		Environment<E> e = (Environment<E>) this.clone();
		e.scopeStack.pop();
		return e;
	}

	public void assoc(EnvironmentEntry<E> entry) {
		List<EnvironmentEntry<E>> scope = scopeStack.peek();
		for (EnvironmentEntry<E> pair : scope) {
			if (pair.getLeft().equals(entry.getLeft()))
				throw new IllegalArgumentException();
		}

		scope.add(entry);
	}

	public E find(String id) {
		ListIterator<List<EnvironmentEntry<E>>> ite = scopeStack.listIterator(scopeStack.size());
		boolean found = false;
		EnvironmentEntry<E> pair = null;
		while (ite.hasPrevious() && !found) {
			List<EnvironmentEntry<E>> scope = ite.previous();
			Iterator<EnvironmentEntry<E>> it = scope.iterator();
			while (it.hasNext() && !found) {
				pair = it.next();
				if (pair.getLeft().equals(id))
					found = true;
			}
		}

		if (found)
			return pair.getRight();
		else
			throw new NoSuchElementException(id + " not declared.");
	}

	public E find(String id, int upDepth) {

		if (upDepth >= this.scopeStack.size())
			throw new NoSuchElementException();

		ListIterator<List<EnvironmentEntry<E>>> ite = scopeStack.listIterator(scopeStack.size());
		boolean found = false;
		EnvironmentEntry<E> pair = null;
		int currDepth = 0;

		while (ite.hasPrevious() && !found && upDepth < this.scopeStack.size() - currDepth){
			List<EnvironmentEntry<E>> scope = ite.previous();
			Iterator<EnvironmentEntry<E>> it = scope.iterator();
			while (it.hasNext() && !found) {
				pair = it.next();
				if (pair.getLeft().equals(id))
				{
					if (upDepth == 0)
						found = true;
					else
						upDepth--;
				}	
			}
			currDepth++;
		}

		if (found)
			return pair.getRight();
		else
			throw new NoSuchElementException(id + " not declared.");
	}

	public int getDepth()
	{
		return scopeStack.size();
	}

	@Override
	@SuppressWarnings("unchecked")
	public Object clone() {
		return new Environment<>((Stack<List<EnvironmentEntry<E>>>) scopeStack.clone());
	}

	private int scopeId;

	public StringBuilder toString(StringBuilder builder)
	{
		scopeId = 0;

		Utils.toStringList(this.scopeStack, 
			(Consumer<List<EnvironmentEntry<E>>>) ((scope) -> 
				{
					builder.append("\nScope " + scopeId + " = ");
					builder.append(scope.toString());
					scopeId++;
				})	, null, Utils.DEFAULT_DELIMITERS, builder);
		return builder;
	}

}