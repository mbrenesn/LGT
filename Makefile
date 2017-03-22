# Upper directory makefile
SHELL=/bin/bash

lgt:
	$(MAKE) -C src/
	mv src/lgt.x .

clean:
	$(MAKE) -C src/ clean
	rm -r *.x 
