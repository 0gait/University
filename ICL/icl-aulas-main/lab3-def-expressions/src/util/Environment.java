package util;

import java.util.*;

public class Environment<E> implements Cloneable {

	private Stack<List<Pair<String, E>>> scopeStack;

	public Environment() {
		scopeStack = new Stack<List<Pair<String, E>>>();
	}

	private Environment(Stack<List<Pair<String, E>>> scopeStack) {
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

	public void assoc(String id, E value) {
		List<Pair<String, E>> scope = scopeStack.peek();
		for (Pair<String, E> pair : scope) {
			if (pair.getLeft().equals(id))
				throw new IllegalArgumentException();
		}

		scope.add(new Pair<String, E>(id, value));
	}

	public E find(String id) {
		ListIterator<List<Pair<String, E>>> ite = scopeStack.listIterator(scopeStack.size());
		boolean found = false;
		Pair<String, E> pair = null;
		while (ite.hasPrevious() && !found) {
			List<Pair<String, E>> scope = ite.previous();
			Iterator<Pair<String, E>> it = scope.iterator();
			while (it.hasNext() && !found) {
				pair = it.next();
				if (pair.getLeft().equals(id))
					found = true;
			}
		}

		if (found)
			return pair.getRight();
		else
			throw new NoSuchElementException();
	}

	public E find(String id, int upDepth) {

		if (upDepth >= this.scopeStack.size())
			throw new NoSuchElementException();

		ListIterator<List<Pair<String, E>>> ite = scopeStack.listIterator(scopeStack.size());
		boolean found = false;
		Pair<String, E> pair = null;
		int currDepth = 0;

		while (ite.hasPrevious() && !found && upDepth < this.scopeStack.size() - currDepth){
			List<Pair<String, E>> scope = ite.previous();
			Iterator<Pair<String, E>> it = scope.iterator();
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
			throw new NoSuchElementException();
	}

	public int getDepth()
	{
		return scopeStack.size();
	}

	@Override
	@SuppressWarnings("unchecked")
	public Object clone() {
		return new Environment<E>((Stack<List<Pair<String, E>>>) scopeStack.clone());
	}

}