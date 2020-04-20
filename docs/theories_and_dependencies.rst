Creating theory classes and dependencies
=========================================

Custom :doc:`theory` can be used to calculate observables needed by a likelihood, or
perform any sub-calculation needed by any other likelihood or theory class. By breaking the calculation
up in to separate classes the calculation can be modularized, and each class can have its own nuisance parameters
and speed, and hence be sampled efficiently using the built-in fast-slow parameter blocking.

Theories should inherit from the base :class:`~.theory.Theory` class, either directly or indirectly.
Each theory may have its own parameters, and depend on derived parameters or general quantities calculated by other
theory classes (or likelihoods as long as this does not lead to a circular dependence).
The names of derived parameters or other quantities are for each class to define and document as needed.

To specify requirements you can use the :class:`~.theory.Theory`  methods

* :meth:`~.theory.Theory.get_requirements`, for things that are always needed.
* return a dictionary from the :meth:`~.theory.Theory.must_provide` method to specify the
  requirements conditional on the quantities that the code is actually being asked to compute


The actual calculation of the quantities requested by the :meth:`~.theory.Theory.must_provide` method should be done by
:meth:`~.theory.Theory.calculate`, which stores computed quantities into a state dictionary. Derived parameters should be
saved into the special ``state['derived']`` dictionary entry.
The theory code also needs to tell other theory codes and likelihoods the things that it can calculate using

*  ``get_X`` methods; any method starting with ``get_`` will automatically indicate that the theory can compute X
* return list of things that can be calculated from  :meth:`~.theory.Theory.get_can_provide`.
* return list of derived parameter names from :meth:`~.theory.Theory.get_can_provide_params`
* specify derived parameters in an associated .yaml file or class params dictionary


Use a ``get_X`` method when you need to add optional arguments to provide different outputs from the computed quantity.
Quantities returned by  :meth:`~.theory.Theory.get_can_provide` should be stored in the state dictionary by the calculate function
or returned by the ``get_results(X)`` for each quantity ``X`` (which by default just returns the value stored in the current state dictionary).
The results stored by calculate for a given set of input parameters are cached, and ``self._current_state`` is set to the current state
whenever ``get_X``, ``get_param`` etc are called.

For example, this is a class that would calculate ``A = B*b_derived`` using inputs ``B`` and derived parameter ``b_derived`` from
another theory code, and provide the method to return ``A`` with a custom normalization and the derived parameter ``Aderived``:

.. code:: python

    from cobaya.theory import Theory

    class ACalculator(Theory):

        def initialize(self):
            """called from __init__ to initialize"""

        def initialize_with_provider(self, provider):
            """
            Initialization after other components initialized, using Provider class
            instance which is used to return any dependencies (see calculate below).
            """
            self.provider = provider

        def get_requirements(self):
            """
            Return dictionary of derived parameters or other quantities that are needed
            by this component and should be calculated by another theory class.
            """
            return {'b_derived': None}

        def must_provide(self, **requirements):
            if 'A' in requirements:
                # e.g. calculating A requires B computed using same kmax (default 10)
                return {'B': {'kmax': requirements['A'].get('kmax', 10)}}

        def get_can_provide_params(self):
            return ['Aderived']

        def calculate(self, state, want_derived=True, **params_values_dict):
            state['A'] = self.provider.get_result('B') * self.provider.get_param('b_derived')
            state['derived'] = {'Aderived': 10}

        def get_A(self, normalization=1):
            return self._current_state['A'] * normalization


Likelihood codes (that return ``A`` in their get_requirements method) can then use,
e.g.  ``self.provider.get_A(normalization=1e-10)`` to get the result calculated by this component.
Some other Theory class would be required to calculate the remaining requirements, e.g.
to get ``b_derived`` and ``B``:

.. code:: python

    from cobaya.theory import Theory

    class BCalculator(Theory):

        def initialize(self):
            self.kmax = 0

        def get_can_provide_params(self):
            return ['b_derived']

        def get_can_provide(self):
            return ['B']

        def must_provide(self, **requirements):
            if 'B' in requirements:
                self.kmax = max(self.kmax, requirements['B'].get('kmax',10))

        def calculate(self, state, want_derived=True, **params_values_dict):
            if self.kmax:
                state['B'] = ... do calculation using self.kmax

            if want_derived:
                state['derived'] = {'b_derived': ...xxx...}


So far this example allows the use of ``ACalculator`` and ``BCalculator`` together with
any likelihood that needs the quantity ``A``, but neither theory code yet depends on any
parameters. Although theory codes do not need to have their own sampled parameters, often
they do, in which case they can be specified in a ``[ClassName].yaml`` file as for
likelihoods, or as a class ``params`` dictionary. For example to specify input parameter
``Xin`` and output parameter ``Xderived`` the class could be defined like this:

.. code:: python

    from cobaya.theory import Theory

    class X(Theory):
        params = {'Xin': None, 'Xderived': {'derived': True}}


Here the user has to specify the input for Xin. Of course you can also provide default
sampling settings for 'Xin' so that configuring it is transparent to the user, e.g.

.. code:: python

    class X(Theory):
        params = {'Xin': {'prior': {'min': 0, 'max': 1}, 'propose': 0.01, 'ref': 0.9},
              'Xderived': {'derived': True}}

If multiple theory codes can provide the same quantity, it may be ambiguous which to use to compute which.
When this happens use the ``provides`` input .yaml keyword to specify that a specific theory computes a
specific quantity.

