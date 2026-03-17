# Static Assets

This directory contains all static assets for the TERRA application.

## Structure

```
static/
├── css/
│   └── styles.css          # Main application styles
├── js/
│   └── app.js             # Main application JavaScript
└── README.md              # This file
```

## Files Description

### CSS Files

- **`css/styles.css`** - Contains all the styling for the TERRA application including:
  - Layout and responsive design
  - Status banner styling
  - Button and form element styles
  - Workflow step indicators
  - Map and sidebar styling
  - Progress bars and status messages

### JavaScript Files

- **`js/app.js`** - Contains all the application logic including:
  - Map initialization and controls
  - Drawing functionality
  - API communication
  - Status management
  - Workflow step management
  - Error handling and user feedback

## Organization Benefits

1. **Separation of Concerns**: HTML, CSS, and JavaScript are now properly separated
2. **Maintainability**: Easier to find and modify specific functionality
3. **Caching**: External files can be cached by browsers for better performance
4. **Reusability**: CSS and JS can be reused across multiple pages if needed
5. **Development**: Easier to work with in development tools and IDEs

## Usage

The files are automatically loaded by Flask using the `url_for()` function:

```html
<!-- CSS -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}" />

<!-- JavaScript -->
<script src="{{ url_for('static', filename='js/app.js') }}"></script>
```

## Future Enhancements

Consider adding:
- `images/` directory for application images
- `fonts/` directory for custom fonts
- `vendor/` directory for third-party libraries
- Minified versions for production
- Source maps for debugging 